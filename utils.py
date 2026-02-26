import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader
import cv2
import time
from collections import defaultdict

import IPython
e = IPython.embed

import glob

# 全局计时器统计
TIMING_STATS = defaultdict(lambda: {"count": 0, "total_time": 0.0})

def log_timing(name, elapsed_time):
    """记录计时统计"""
    TIMING_STATS[name]["count"] += 1
    TIMING_STATS[name]["total_time"] += elapsed_time

def print_timing_stats():
    """打印计时统计信息"""
    print("\n" + "="*60)
    print("⏱️  性能分析统计")
    print("="*60)
    sorted_stats = sorted(TIMING_STATS.items(), key=lambda x: x[1]["total_time"], reverse=True)
    for name, stats in sorted_stats:
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        print(f"{name:40s} | 总计: {stats['total_time']:8.2f}s | "
              f"调用: {stats['count']:6d}次 | 平均: {avg_time*1000:7.2f}ms")
    print("="*60 + "\n")

def find_all_hdf5(dataset_dir):
    """
    递归查找 dataset_dir 下所有文件名包含 'recovery' 的 .h5 文件
    """
    patterns = [
        os.path.join(dataset_dir, '**', '*.h5'),
        os.path.join(dataset_dir, '**', '*.hdf5')
    ]
    all_paths = []
    for p in patterns:
        all_paths.extend(glob.glob(p, recursive=True))
    
    # === 新增：过滤逻辑 ===
    # 只保留文件名中包含 "recovery" 的文件
    filtered_paths = [p for p in all_paths if 'recovery' in os.path.basename(p)]
    
    filtered_paths.sort()
    
    if len(filtered_paths) == 0:
        # 为了防止报错，打印一下原始找到了多少文件，方便排查
        print(f"Debug: Found {len(all_paths)} total files, but 0 match 'recovery'.")
        raise ValueError(f"No 'recovery' HDF5 files found in {dataset_dir}")
        
    print(f"Found {len(filtered_paths)} 'recovery' episodes in {dataset_dir}")
    return filtered_paths

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, camera_names, norm_stats, loader_mode='full_episode', chunk_size=50, stride=1, random_start=True, preload_to_memory=False):
        super(EpisodicDataset).__init__()
        self.file_paths = file_paths
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.loader_mode = loader_mode
        self.chunk_size = int(chunk_size)
        self.stride = int(stride)
        self.random_start = random_start
        self.preload_to_memory = bool(preload_to_memory)
        self.is_sim = None
        self.sample_index = []
        self.episodes_cache = []
        self.episode_lengths = []
        self.action_mean_t = torch.as_tensor(self.norm_stats["action_mean"], dtype=torch.float32)
        self.action_std_t = torch.as_tensor(self.norm_stats["action_std"], dtype=torch.float32)
        self.qpos_mean_t = torch.as_tensor(self.norm_stats["qpos_mean"], dtype=torch.float32)
        self.qpos_std_t = torch.as_tensor(self.norm_stats["qpos_std"], dtype=torch.float32)

        if self.loader_mode not in ['full_episode', 'sliding_window']:
            raise ValueError(f"Invalid loader_mode={self.loader_mode}. Use 'full_episode' or 'sliding_window'.")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")

        if len(self.file_paths) == 0:
            raise ValueError("No file_paths provided to EpisodicDataset.")

        if self.preload_to_memory:
            self._preload_all_episodes()
        else:
            self._index_from_files()

        if self.loader_mode == 'sliding_window':
            print(f"Indexing {len(self.file_paths)} episodes... (mode={self.loader_mode}, chunk={self.chunk_size}, stride={self.stride})")
            for file_idx, episode_len in enumerate(self.episode_lengths):
                for start_ts in range(0, episode_len, self.stride):
                    self.sample_index.append((file_idx, start_ts))
            print(f"Total samples indexed: {len(self.sample_index)}")

        self.__getitem__(0)  # initialize self.is_sim

    def _index_from_files(self):
        self.episode_lengths = []
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as root:
                self.episode_lengths.append(root['/action/joint_positions'].shape[0])

    def _preload_all_episodes(self):
        print(f"Preloading {len(self.file_paths)} episodes into RAM...")
        q_left_index = [0, 2, 4, 6, 8, 10, 12]
        self.episodes_cache = []
        self.episode_lengths = []
        total_frames = 0
        for file_idx, dataset_path in enumerate(self.file_paths):
            with h5py.File(dataset_path, 'r') as root:
                self.is_sim = True
                episode_len = root['/action/joint_positions'].shape[0]
                self.episode_lengths.append(episode_len)
                total_frames += episode_len

                base_pose = root['/observations/base_pose'][()]
                base_pos = base_pose[:, :3]
                base_vel = root['/observations/base_vel'][()]
                full_qos = root['/observations/joint_pos'][()]
                qpos_gripper = np.zeros((episode_len, 1), dtype=np.float32)
                joint_qpos = np.concatenate([full_qos[:, q_left_index], qpos_gripper], axis=1)
                qpos_seq = np.concatenate([joint_qpos, base_pos, base_vel], axis=1).astype(np.float32)
                qpos_seq_t = torch.from_numpy(qpos_seq)
                qpos_seq_norm = (qpos_seq_t - self.qpos_mean_t) / self.qpos_std_t

                cam_image_seqs = []
                for cam_name in self.camera_names:
                    cam_imgs = []
                    for t in range(episode_len):
                        img_bytes = root[f'/observations/{cam_name}'][t]
                        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        cam_imgs.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    cam_image_seqs.append(np.stack(cam_imgs, axis=0))
                # (T, K, H, W, C) -> (T, K, C, H, W), keep uint8 to save RAM.
                image_seq = np.stack(cam_image_seqs, axis=1)
                image_seq_t = torch.from_numpy(image_seq).permute(0, 1, 4, 2, 3).contiguous()

                action_joints_full = root['/action/joint_positions'][()]
                action_joints = action_joints_full[:, q_left_index]
                action_joints_vel_full = root['/action/joint_velocities'][()]
                action_joints_vel = action_joints_vel_full[:, q_left_index]
                action_gripper_full = root['/action/gripper_command'][()]
                action_gripper = action_gripper_full[:, 0:1]
                action_gripper_vel = action_gripper_full[:, 0:1]
                action_seq = np.concatenate(
                    [action_joints, action_gripper, action_joints_vel, action_gripper_vel], axis=-1
                ).astype(np.float32)
                action_seq_t = torch.from_numpy(action_seq)
                action_seq_norm = (action_seq_t - self.action_mean_t) / self.action_std_t

                self.episodes_cache.append(
                    {
                        "qpos_seq_norm": qpos_seq_norm,
                        "image_seq": image_seq_t,
                        "action_seq_norm": action_seq_norm,
                        "episode_len": episode_len,
                    }
                )
            if (file_idx + 1) % 20 == 0 or file_idx + 1 == len(self.file_paths):
                print(f"  Preloaded {file_idx + 1}/{len(self.file_paths)} episodes")
        print(f"Preload done. Total frames: {total_frames}")

    def __len__(self):
        if self.loader_mode == 'sliding_window':
            return len(self.sample_index)
        return len(self.file_paths)

    def __getitem__(self, index):
        if self.loader_mode == 'sliding_window':
            file_idx, start_ts = self.sample_index[index]
        else:
            file_idx = index
            start_ts = None

        if self.preload_to_memory:
            episode = self.episodes_cache[file_idx]
            episode_len = episode["episode_len"]
            if start_ts is None:
                if self.random_start:
                    start_ts = np.random.randint(episode_len)
                else:
                    start_ts = 0
            qpos_data = episode["qpos_seq_norm"][start_ts]
            image_data = episode["image_seq"][start_ts].float() / 255.0
            action_seq_norm = episode["action_seq_norm"][start_ts:]
            if self.loader_mode == 'sliding_window':
                target_len = self.chunk_size
            else:
                target_len = episode_len
            action_len = min(action_seq_norm.shape[0], target_len)
            action_data = torch.zeros((target_len, action_seq_norm.shape[-1]), dtype=torch.float32)
            if action_len > 0:
                action_data[:action_len] = action_seq_norm[:action_len]
            is_pad = torch.ones(target_len, dtype=torch.bool)
            is_pad[:action_len] = False
            return image_data, qpos_data, action_data, is_pad
        else:
            dataset_path = self.file_paths[file_idx]
            with h5py.File(dataset_path, 'r') as root:
                self.is_sim = True
                original_action_shape = root['/action/joint_positions'].shape
                episode_len = original_action_shape[0]

                if start_ts is None:
                    if self.random_start:
                        start_ts = np.random.randint(episode_len)
                    else:
                        start_ts = 0

                base_pose = root['/observations/base_pose'][start_ts]
                base_pos = base_pose[:3]
                base_vel = root['/observations/base_vel'][start_ts]
                full_qos = root['/observations/joint_pos'][start_ts]
                q_left_index = [0,2,4,6,8,10,12]
                qpos_gripper = np.array([0.0])
                joint_qpos = np.concatenate([full_qos[q_left_index], qpos_gripper])
                qpos = np.concatenate([joint_qpos, base_pos, base_vel])

                image_dict = dict()
                for cam_name in self.camera_names:
                    img_bytes = root[f'/observations/{cam_name}'][start_ts]
                    img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    image_dict[cam_name] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                action_joints_full = root['/action/joint_positions'][start_ts:]
                action_joints = action_joints_full[:, q_left_index]
                action_joints_vel_full = root['/action/joint_velocities'][start_ts:]
                action_joints_vel = action_joints_vel_full[:, q_left_index]
                action_gripper_full = root['/action/gripper_command'][start_ts:]
                action_gripper = action_gripper_full[:, 0:1]
                action_gripper_vel = action_gripper_full[:, 0:1]
                action = np.concatenate([action_joints, action_gripper, action_joints_vel, action_gripper_vel], axis=-1)

                if self.loader_mode == 'sliding_window':
                    target_len = self.chunk_size
                else:
                    target_len = original_action_shape[0]

        # non-preload branch only
        action_len = min(action.shape[0], target_len)

        padded_action = np.zeros((target_len, action.shape[-1]), dtype=np.float32)
        padded_action[:action_len] = action[:action_len]
        is_pad = np.zeros(target_len, dtype=np.float32)
        is_pad[action_len:] = 1

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum('k h w c -> k c h w', image_data)

        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad

def get_norm_stats(file_paths, num_episodes): 
    all_qpos_data = []
    all_action_data = []
    
    # 如果指定了 num_episodes，只计算前 N 个文件的统计信息（节省时间）
    # 或者你可以选择随机采样 N 个
    paths_to_process = file_paths[:num_episodes] if num_episodes < len(file_paths) else file_paths

    print(f"Calculating stats from {len(paths_to_process)} episodes...")

    for file_path in paths_to_process:
        with h5py.File(file_path, 'r') as root:
            # 读取基座位置和速度
            base_pose = root['/observations/base_pose'][()]  # (T, 7)
            base_pos = base_pose[:, :3]  # (T, 3) 只取位置
            base_vel = root['/observations/base_vel'][()]  # (T, 6)
            
            # 读取关节位置
            full_qpos = root['/observations/joint_pos'][()]
            q_left_index = [0,2,4,6,8,10,12]  # 左臂关节索引，后续7个是夹爪
            q_pos_gripper = np.zeros((full_qpos.shape[0], 1))
            joint_qpos = np.concatenate([full_qpos[:, q_left_index], q_pos_gripper], axis=1)
            
            # 组合: 关节位置(8) + 基座位置(3) + 基座速度(6) = 17维
            qpos = np.concatenate([joint_qpos, base_pos, base_vel], axis=1)

            # 位置 (Action)
            action_joints_pos_full = root['/action/joint_positions'][()]
            action_joints_pos = action_joints_pos_full[:, q_left_index]  # (T, 7)
            action_joints_vel_full = root['/action/joint_velocities'][()]
            action_joints_vel = action_joints_vel_full[:, q_left_index]  # (T, 7)
            action_gripper_full = root['/action/gripper_command'][()]
            action_gripper = action_gripper_full[:, 0:1]  # 只取左臂夹爪 (T, 1)
            action_gripper_vel = action_gripper_full[:, 0:1]  # 假设夹爪速度为0
            
            # 组合: joint_pos(7) + gripper(1) = 8 维
            action = np.concatenate([action_joints_pos, action_gripper, action_joints_vel, action_gripper_vel ], axis=-1)
            
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    
    # 堆叠所有数据
    all_qpos_data = torch.cat(all_qpos_data, dim=0) # 注意这里用 cat 而不是 stack，把所有时间步拼在一起
    all_action_data = torch.cat(all_action_data, dim=0)

    # 计算均值和方差
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # 防止除零

    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze()+1e-6,
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze()+1e-6,
        "example_qpos": qpos # 只是为了调试，可以留着
    }

    return stats


def load_mixed_data(data_root, num_episodes, camera_names, batch_size_train, batch_size_val, loader_mode='full_episode', chunk_size=50, stride=1, num_workers=1, preload_to_memory=False):
    """
    随机在 data_ik 和 data_ja 下各抽取或随机抽取两个子文件夹，
    并从每个文件夹中随机抽取 50% 的文件进行混合训练。
    使用回档后的加载逻辑。
    """
    import random
    print(f'\nLoading Mixed Data from: {data_root}')
    
    # 1. 搜集所有子文件夹 (例如 data/data_ik/aura_k1)
    subfolders = []
    for method in ['data_ik', 'data_ja']:
        method_path = os.path.join(data_root, method)
        if os.path.exists(method_path):
            current_subs = [os.path.join(method_path, f) for f in os.listdir(method_path) 
                           if os.path.isdir(os.path.join(method_path, f))]
            subfolders.extend(current_subs)
    
    if len(subfolders) == 0:
        raise ValueError(f"No subfolders found in {data_root}.")

    # 2. 使用所有子文件夹
    selected_folders = subfolders
    print(f"Selected {len(selected_folders)} folders for mixed training:")
    for folder in selected_folders:
        print(f"  - {folder}")

    # 3. 从所有文件夹中搜集所有文件
    all_mixed_paths = []
    for folder in selected_folders:
        folder_files = find_all_hdf5(folder)
        print(f"  Folder: {os.path.basename(folder)} | Found {len(folder_files)} recovery files")
        all_mixed_paths.extend(folder_files)

    total_files_before = len(all_mixed_paths)
    print(f"Total files from both folders: {total_files_before}")
    
    # 4. 从所有文件中随机抽取50% (即450条，如果总共900条)
    random.shuffle(all_mixed_paths)
    keep_count = max(1, total_files_before // 2)
    all_mixed_paths = all_mixed_paths[:keep_count]
    total_files = len(all_mixed_paths)
    print(f"Selected {total_files} files (50% of total) for mixed training")

    # 5. 后续逻辑与 load_data 一致：打乱、划分、计算 stats
    shuffled_indices = np.random.permutation(total_files)
    shuffled_paths = [all_mixed_paths[i] for i in shuffled_indices]
    
    train_ratio = 0.8
    num_train = int(train_ratio * total_files)
    train_paths = shuffled_paths[:num_train]
    val_paths = shuffled_paths[num_train:]

    print(f"Train files: {len(train_paths)}, Val files: {len(val_paths)}")

    # 计算统计值
    norm_stats = get_norm_stats(train_paths, num_episodes=num_episodes)

    if preload_to_memory and os.name == 'nt' and num_workers > 0:
        print(f"[WARN] preload_to_memory=True with num_workers={num_workers} on Windows may duplicate RAM per worker.")
        print("[WARN] Consider setting --dataloader_workers 0 or 1 when preloading to memory.")

    train_dataset = EpisodicDataset(
        train_paths, camera_names, norm_stats,
        loader_mode=loader_mode, chunk_size=chunk_size, stride=stride, random_start=(loader_mode == 'full_episode'),
        preload_to_memory=preload_to_memory
    )
    val_dataset = EpisodicDataset(
        val_paths, camera_names, norm_stats,
        loader_mode=loader_mode, chunk_size=chunk_size, stride=stride, random_start=False,
        preload_to_memory=preload_to_memory
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=num_workers)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, loader_mode='full_episode', chunk_size=50, stride=1, num_workers=1, preload_to_memory=False):
    print(f'\nData from: {dataset_dir}\n')
    
    # 1. 获取所有文件的绝对路径 (包含子文件夹)
    all_file_paths = find_all_hdf5(dataset_dir)
    total_files = len(all_file_paths)
    
    # 如果 num_episodes 只是用来限制数据量，可以在这里截断
    # 如果 num_episodes 是指“用多少数据计算norm”，则在 get_norm_stats 里处理
    # 这里我们假设使用所有找到的文件
    
    # 2. 打乱路径列表
    shuffled_indices = np.random.permutation(total_files)
    shuffled_paths = [all_file_paths[i] for i in shuffled_indices] # 使用列表推导式重排
    
    # 3. 划分训练/验证集
    train_ratio = 0.8
    num_train = int(train_ratio * total_files)
    
    train_paths = shuffled_paths[:num_train]
    val_paths = shuffled_paths[num_train:]

    print(f"Train files: {len(train_paths)}, Val files: {len(val_paths)}")

    # 4. 计算统计值 (传入路径列表)
    # 注意：这里我们通常用所有训练数据来计算 Stats，或者取前 num_episodes 个
    norm_stats = get_norm_stats(train_paths, num_episodes=num_episodes)

    if preload_to_memory and os.name == 'nt' and num_workers > 0:
        print(f"[WARN] preload_to_memory=True with num_workers={num_workers} on Windows may duplicate RAM per worker.")
        print("[WARN] Consider setting --dataloader_workers 0 or 1 when preloading to memory.")

    train_dataset = EpisodicDataset(
        train_paths, camera_names, norm_stats,
        loader_mode=loader_mode, chunk_size=chunk_size, stride=stride, random_start=(loader_mode == 'full_episode'),
        preload_to_memory=preload_to_memory
    )
    val_dataset = EpisodicDataset(
        val_paths, camera_names, norm_stats,
        loader_mode=loader_mode, chunk_size=chunk_size, stride=stride, random_start=False,
        preload_to_memory=preload_to_memory
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=num_workers)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim



### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
