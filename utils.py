import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import cv2
import time
from collections import defaultdict

import IPython
e = IPython.embed

import glob
from functools import lru_cache

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

@lru_cache(maxsize=1000) # 缓存最近使用的 1000 帧解码图像

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
    def __init__(self, file_paths, camera_names, norm_stats, num_obs, num_action):
        super(EpisodicDataset, self).__init__()
        self.num_obs = num_obs
        self.num_action = num_action
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = True
        
        # === 核心优化：内存数据仓库 ===
        # 我们不存 h5py 对象，而是把数据读入 RAM
        # 但对于图像，我们只存 "bytes" (压缩数据)，不解码，以节省内存
        self.episode_data = [] 
        self.samples = [] 
        
        print(f"Loading {len(file_paths)} episodes to RAM (Images kept as bytes)...")
        
        q_left_index = [0, 2, 4, 6, 8, 10, 12] # 左臂关节索引

        for ep_i, path in enumerate(file_paths):
            with h5py.File(path, 'r') as root:
                # 1. 读取基础数据
                # 注意：这里我们直接处理好 qpos 和 action 的维度，避免在 getitem 里重复切片
                
                # --- QPOS 处理 (21维) ---
                base_pose = root['/observations/base_pose'][()]
                base_vel = root['/observations/base_vel'][()]
                full_qpos = root['/observations/joint_pos'][()]
                T = full_qpos.shape[0]
                
                # 构造 joint_qpos (8维: 7关节 + 1夹爪)
                # 假设 sim 数据中 joint_pos 是 14 维，我们需要取左臂
                # 如果你的数据结构不同，请检查这里的维度
                joint_qpos_left = full_qpos[:, q_left_index]
                # 模拟数据通常夹爪需要单独处理，这里假设追加一个 0 或者从某处读取
                # 你的原始代码中是: qpos_gripper = np.array([0.0])，这里我们在 batch 维度构造
                gripper_pad = np.zeros((T, 1), dtype=np.float32) 
                
                qpos_data = np.concatenate([joint_qpos_left, gripper_pad, base_pose, base_vel], axis=1).astype(np.float32)

                # --- Action 处理 (16维) ---
                action_pos_full = root['/action/joint_positions'][()]
                action_vel_full = root['/action/joint_velocities'][()]
                action_gripper_full = root['/action/gripper_command'][()]
                
                action_pos = action_pos_full[:, q_left_index]
                action_vel = action_vel_full[:, q_left_index]
                action_gripper = action_gripper_full[:, 0:1] # 取第一维
                action_gripper_vel = np.zeros((T, 1), dtype=np.float32)
                
                action_data = np.concatenate([action_pos, action_gripper, action_vel, action_gripper_vel], axis=1).astype(np.float32)

                # --- 图像 Bytes 读取 ---
                # 关键：只读 bytes，不解码
                image_bytes_dict = {}
                for cam in camera_names:
                    # h5py 读取出的通常是 numpy 数组格式的 bytes
                    image_bytes_dict[cam] = root[f'observations/{cam}'][()]

                # 存入列表
                self.episode_data.append({
                    'qpos': qpos_data,
                    'action': action_data,
                    'images': image_bytes_dict,
                    'len': T
                })

                # --- 建立索引 ---
                # 遍历 T-1 步 (根据你的逻辑)
                for t in range(T - 1):
                    self.samples.append((ep_i, t))

        print(f"Loaded {len(self.samples)} samples. Ready for training.")

    @staticmethod
    @lru_cache(maxsize=3000) # 缓存 3000 张图，足够覆盖 num_workers * batch_size * num_obs
    def _cached_decode(img_bytes_raw):
        """
        静态方法 + LRU 缓存
        必须接收 bytes 类型 (hashable)，不能接收 numpy array
        """
        nparr = np.frombuffer(img_bytes_raw, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ep_idx, cur_idx = self.samples[index]
        data = self.episode_data[ep_idx]
        T = data['len']

        # 1. 确定 Observation 窗口索引
        # 如果 cur_idx < num_obs, 用第一帧填充
        obs_ids = []
        for i in range(self.num_obs):
            t = max(0, cur_idx - (self.num_obs - 1) + i)
            obs_ids.append(t)

        # 2. 确定 Action 预测窗口索引
        act_start = cur_idx # 包含当前帧
        action_ids = []
        is_pad = np.zeros(self.num_action, dtype=bool)
        for i in range(self.num_action):
            t = act_start + i
            if t >= T:
                t = T - 1
                is_pad[i] = True
            action_ids.append(t)

        # 3. 获取 QPOS 和 Action (直接切片 RAM 中的数组，极快)
        qpos_data = torch.from_numpy(data['qpos'][obs_ids]).float()
        action_data = torch.from_numpy(data['action'][action_ids]).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # 归一化
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # 4. 获取图像 (解码 + 缓存)
        all_imgs = []
        for cam in self.camera_names:
            cam_imgs = []
            # 获取该相机该 episode 的所有 bytes 数组
            full_bytes_seq = data['images'][cam] 
            
            for t in obs_ids:
                # 取出单帧的 bytes numpy
                img_byte_np = full_bytes_seq[t]
                # 关键：转为 python bytes 传入缓存函数
                img = self._cached_decode(img_byte_np.tobytes())
                cam_imgs.append(img)
            
            all_imgs.append(np.stack(cam_imgs)) # (num_obs, H, W, 3)

        # 堆叠 -> (num_obs, num_cam, H, W, 3)
        image_data = np.stack(all_imgs, axis=1) 
        
        # 转 Tensor 并调整维度 -> (num_obs, num_cam, 3, H, W)
        image_data = torch.from_numpy(image_data).float() / 255.0
        image_data = torch.einsum('t k h w c -> t k c h w', image_data)

        # 兼容性处理：如果 num_obs=1，去掉时间维度
        if self.num_obs == 1:
            image_data = image_data.squeeze(0)
            qpos_data = qpos_data.squeeze(0)

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
            
            # 组合:关节位置(8)+ 基座位置(3) + 基座姿态(4) + 基座速度(6) +  = 21维
            qpos = np.concatenate([joint_qpos, base_pose, base_vel], axis=1)

            # 位置
            action_joints_pos_full = root['/action/joint_positions'][()]
            action_joints_pos = action_joints_pos_full[:, q_left_index]  # (T, 7)
            action_gripper_full = root['/action/gripper_command'][()]
            action_gripper = action_gripper_full[:, 0:1]  # 只取左臂夹爪 (T, 1)
            
            # 速度
            action_joints_vel_full = root['/action/joint_velocities'][()]
            action_joints_vel = action_joints_vel_full[:, q_left_index]  # (T, 7)
            action_gripper_vel = np.zeros((action_gripper.shape[0], 1), dtype=np.float32)  # (T, 1)
            
            # 组合: joint_pos(7) + gripper(1) + joint_vel(7) + gripper_vel(1) = 16
            action = np.concatenate([action_joints_pos, action_gripper,
                                    action_joints_vel, action_gripper_vel], axis=-1)
            
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


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size=50, num_obs=1):
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
    train_ratio = 0.9  # 增加训练集比例，减少验证集（从0.8改为0.9）
    num_train = int(train_ratio * total_files)
    
    train_paths = shuffled_paths[:num_train]
    val_paths = shuffled_paths[num_train:]

    print(f"Train files: {len(train_paths)}, Val files: {len(val_paths)}")

    # 4. 计算统计值 (传入路径列表)
    # 注意：这里我们通常用所有训练数据来计算 Stats，或者取前 num_episodes 个
    norm_stats = get_norm_stats(train_paths, num_episodes=num_episodes)

    # 5. 构造 Dataset (传入路径列表)
    train_dataset = EpisodicDataset(train_paths, camera_names, norm_stats, num_obs=num_obs, num_action=chunk_size)
    val_dataset = EpisodicDataset(val_paths, camera_names, norm_stats, num_obs=num_obs, num_action=chunk_size)
    
    # 使用单进程加载以避免多进程内存问题
    # 如果数据加载太慢，考虑预加载或使用更高效的数据格式
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, pin_memory=False, num_workers=0)

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
