import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import cv2

import IPython
e = IPython.embed

import glob
from functools import lru_cache
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
        self.file_paths = file_paths
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_obs = num_obs
        self.num_action = num_action
        self.is_sim = True # 默认为仿真
        # --- 新增：扫描所有轨迹，生成样本索引映射 ---
        self.samples = []
        self.indices = []
        self.files_cache = {} # 文件句柄缓存
        print(f"Indexing {len(file_paths)} episodes...")
        for path in file_paths:
            with h5py.File(path, 'r') as root:
                # 获取该轨迹的总步数 T
                T = root['/action/joint_positions'].shape[0]
                # 参考 space_dataset 逻辑：每个 cur_idx 作为一个样本的“当前步”
                # 如果要预测下一步开始的动作，通常 cur_idx 遍历到 T-2
                for cur_idx in range(T - 1):
                    self.samples.append((path, cur_idx))
        print(f"Total samples indexed: {len(self.samples)}")
    

    def __len__(self):
        return len(self.samples) # <--- 修改这里
    
    @lru_cache(maxsize=1000) # 缓存最近使用的 1000 帧解码图像
    def _decode_image(img_bytes):
        return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 在 __getitem__ 中调用
    
    def __getitem__(self, index):
        file_path, cur_idx = self.samples[index]
        # 2. 优化：文件句柄缓存 (LRU Cache 简单版)
        if file_path not in self.files_cache:
            # 这里的 swmr=True 很重要，允许单写多读，且更稳定
            self.files_cache[file_path] = h5py.File(file_path, 'r', swmr=True, libver='latest')
        
        root = self.files_cache[file_path]
        # with h5py.File(file_path, 'r') as root:
            # 1. 确定 Observation 索引 (观测窗口)
            # 如果 cur_idx 不够取 num_obs 帧，则用第一帧填充前面
        obs_ids = []
        for i in range(self.num_obs):
            t = max(0, cur_idx - (self.num_obs - 1) + i)
            obs_ids.append(t)

        # 2. 确定 Action 索引 (预测窗口)
        # 从 cur_idx 或 cur_idx + 1 开始取未来 num_action 帧
        # 如果后面步数不够，用最后一帧填充
        act_start = cur_idx # 如果你想预测包含当前帧开始，设为 cur_idx；如果从下一帧开始，设为 cur_idx + 1
        action_ids = []
        T = root['/action/joint_positions'].shape[0]
        for i in range(self.num_action):
            t = min(T - 1, act_start + i)
            action_ids.append(t)
            
        # 标记哪些 action 分步是填充的（Padding Mask）
        is_pad = np.zeros(self.num_action, dtype=bool)
        for i, t in enumerate(action_ids):
            if act_start + i >= T:
                is_pad[i] = True

        # 3. 读取并处理 QPOS (观测状态)
        # 我们取观测窗口内每一帧的状态
        q_left_index = [0,2,4,6,8,10,12]
        all_qpos = []
        for t in obs_ids:
            base_pose = root['/observations/base_pose'][t]
            base_pos = base_pose[:3]
            base_vel = root['/observations/base_vel'][t]
            full_qos = root['/observations/joint_pos'][t]
            qpos_gripper = np.array([0.0])
            joint_qpos = np.concatenate([full_qos[q_left_index], qpos_gripper])
            qpos_t = np.concatenate([joint_qpos, base_pos, base_pose[3:7], base_vel])
            all_qpos.append(qpos_t)
        qpos_data = np.stack(all_qpos, axis=0) # (num_obs, state_dim)

        # 4. 读取图像
        # 结果形状: (num_obs, num_cam, C, H, W)
        # 读取 images (最耗时部分)
        all_imgs = []
        for cam in self.camera_names:
            # 优化：只读取需要的帧
            # h5py 读取 bytes 是很快的，耗时在 decode
            img_bytes_seq = root[f'observations/{cam}'][obs_ids]  # (num_obs, )
            
            cam_imgs = []
            for img_bytes in img_bytes_seq:
                img = self._decode_image(img_bytes)
                cam_imgs.append(img)
                #img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                #cam_imgs.append(img)
            all_imgs.append(np.stack(cam_imgs)) # (num_obs, H, W, 3)
            

        image_data = np.stack(all_imgs, axis=1) # (num_obs, num_cam, H, W, 3)

        # 5. 读取动作 Action
        # 先用 [:] 读取完整数组，避免 h5py fancy indexing 的 increasing order 限制
        action_joints_pos_full = root['/action/joint_positions'][:]  # (T, 14)
        action_gripper_full = root['/action/gripper_command'][:]      # (T, 2)
        action_joints_vel_full = root['/action/joint_velocities'][:]  # (T, 14)
            
        # 再用列表索引
        action_joints_pos = action_joints_pos_full[action_ids][:, q_left_index]  # (num_action, 7)
        action_gripper = action_gripper_full[action_ids][:, 0:1]                  # (num_action, 1)
        action_joints_vel = action_joints_vel_full[action_ids][:, q_left_index]  # (num_action, 7)
        action_gripper_vel = np.zeros((len(action_ids), 1), dtype=np.float32)    # (num_action, 1)
            
        action_data = np.concatenate([action_joints_pos, action_gripper, 
                                     action_joints_vel, action_gripper_vel], axis=-1)

        # 6. 转为 Tensor 并归一化
        image_data = torch.from_numpy(image_data).float() / 255.0
        # 调整图像维度: (num_obs, num_cam, H, W, 3) -> (num_obs, num_cam, 3, H, W)
        image_data = torch.einsum('t k h w c -> t k c h w', image_data)
        
        qpos_data = torch.from_numpy(qpos_data).float()
        action_data = torch.from_numpy(action_data).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # 归一化
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # 如果 num_obs == 1，为了兼容原来的模型，squeeze 掉 num_obs 维度
        if self.num_obs == 1:
            image_data = image_data.squeeze(0)  # (num_cam, 3, H, W)
            qpos_data = qpos_data.squeeze(0)    # (state_dim,)
        
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
