import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import cv2

import IPython
e = IPython.embed

import glob

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
    def __init__(self, file_paths, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.file_paths = file_paths # <--- 保存路径列表，而不是 IDs
        #self.episode_ids = episode_ids
        
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.file_paths) # <--- 修改这里

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        dataset_path = self.file_paths[index] # <--- 修改这里
        # 兼容你的文件名格式: episode_{seq}_{id}_{type}_{light}.h5
        # 如果你无法确定具体文件名，建议用 glob 搜索，这里假设你已经重命名或者格式固定
        # 简单起见，这里假设文件名就是 episode_0.h5, episode_1.h5... 
        # 如果不是，你需要修改这里的文件名匹配逻辑
        # 目前只加载recovery的episode

        with h5py.File(dataset_path, 'r') as root:
            # 1. 模拟状态判断 (根据你的metadata结构调整)
            # 你的代码里把它放在了 metadata 组里，或者 attrs 里
            self.is_sim = True # AstroBench 肯定是仿真，直接由 True 即可
            
            # 2. 获取 Action 长度
            original_action_shape = root['/action/joint_positions'].shape
            episode_len = original_action_shape[0]
            
            # 3. 确定采样时间点
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            
            # 4. 获取基座位置和速度
            base_pose = root['/observations/base_pose'][start_ts]  # (7,) [pos(3) + quat(4)]
            base_pos = base_pose[:3]  # 只取位置 (3,)
            base_vel = root['/observations/base_vel'][start_ts]  # (6,) [linear(3) + angular(3)]
            
            # 5. 获取 QPOS (当前机器人状态)
            # 你的数据里是 joint_pos (28维), 通常 ACT 需要包含夹爪状态
            # 如果 observation 里没有存夹爪状态，我们暂时只取关节
            full_qos = root['/observations/joint_pos'][start_ts] # (28,)
            q_left_index = [0,2,4,6,8,10,12]  # 左臂关节索引，后续7个是夹爪
            q_right_index = [1,3,5,7,9,11,13] # 右臂关节索引
            qpos_gripper = np.array([0.0]) 
            joint_qpos = np.concatenate([full_qos[q_left_index], qpos_gripper])  # 8 维
            
            # 组合: 基座位置(3) + 基座速度(6) + 关节位置(8) = 17维
            qpos = np.concatenate([base_pos, base_vel, joint_qpos])
            
            # 6. 获取 QVEL (速度) - 暂时不使用，保留接口
            qvel = np.zeros_like(qpos) 

            # 7. 获取图像 (核心修改: 解码 JPEG/PNG)
            image_dict = dict()
            for cam_name in self.camera_names:
                # 读取字节流
                img_bytes = root[f'/observations/{cam_name}'][start_ts]
                # 解码: bytes -> numpy (BGR)
                img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                # 转换: BGR -> RGB
                image_dict[cam_name] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 8. 获取动作 Action (合并关节位置、速度和夹爪)
            # 位置
            action_joints_pos_full = root['/action/joint_positions'][start_ts:]
            action_joints_pos = action_joints_pos_full[:, q_left_index]  # (T, 7)
            action_gripper_full = root['/action/gripper_command'][start_ts:]
            action_gripper = action_gripper_full[:, 0:1]  # 只取左臂夹爪 (T, 1)
            
            # 速度
            action_joints_vel_full = root['/action/joint_velocities'][start_ts:]
            action_joints_vel = action_joints_vel_full[:, q_left_index]  # (T, 7)
            # 夹爪速度（如果没有单独的夹爪速度数据，可以设为0或从gripper_command计算差分）
            action_gripper_vel = np.zeros((action_gripper.shape[0], 1), dtype=np.float32)  # (T, 1)
            
            # 组合: joint_pos(7) + gripper(1) + joint_vel(7) + gripper_vel(1) = 16
            action = np.concatenate([action_joints_pos, action_gripper, 
                                    action_joints_vel, action_gripper_vel], axis=-1)  # (T, 16)
            
            action_len = episode_len - start_ts

        # 构造 Padding
        # 注意: action.shape[-1] 应该是 16 (7关节位置 + 1夹爪 + 7关节速度 + 1夹爪速度)
        padded_action = np.zeros((original_action_shape[0], action.shape[-1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # 堆叠图像
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # 构造 PyTorch Tensor
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Channel Last -> Channel First
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # 归一化 (使用传入的 stats)
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(file_paths, num_episodes): # <--- 接收路径列表
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
            
            # 组合: 基座位置(3) + 基座速度(6) + 关节位置(8) = 17维
            qpos = np.concatenate([base_pos, base_vel, joint_qpos], axis=1)

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
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos # 只是为了调试，可以留着
    }

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
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

    # 5. 构造 Dataset (传入路径列表)
    train_dataset = EpisodicDataset(train_paths, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_paths, camera_names, norm_stats)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1)

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
