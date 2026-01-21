import numpy as np
import torch
import os
import h5py
import glob

# ================= 配置区域 =================
# 请确保这里改为你真实的绝对路径
DATASET_DIR = '/home/gpuserver/hx/github/act/data' 
NUM_EPISODES_TO_PROCESS = 50  # 计算多少个文件
# ===========================================

def find_all_hdf5(dataset_dir):
    """递归查找所有带 'recovery' 的 .h5 文件"""
    patterns = [
        os.path.join(dataset_dir, '**', '*.h5'),
        os.path.join(dataset_dir, '**', '*.hdf5')
    ]
    all_paths = []
    for p in patterns:
        all_paths.extend(glob.glob(p, recursive=True))
    
    # 过滤：只保留文件名包含 'recovery' 的
    filtered_paths = [p for p in all_paths if 'recovery' in os.path.basename(p)]
    filtered_paths.sort()
    
    if len(filtered_paths) == 0:
        print(f"DEBUG: 在 {dataset_dir} 找到了 {len(all_paths)} 个文件，但没有包含 'recovery' 的文件。")
        raise ValueError("No 'recovery' files found!")
        
    print(f"找到了 {len(filtered_paths)} 个 recovery 文件。")
    return filtered_paths

def get_norm_stats(file_paths, num_episodes):
    all_qpos_data = []
    all_action_data = []
    
    # 限制处理数量
    paths_to_process = file_paths[:num_episodes] if num_episodes < len(file_paths) else file_paths
    print(f"正在读取 {len(paths_to_process)} 个文件进行计算...")

    for file_path in paths_to_process:
        try:
            with h5py.File(file_path, 'r') as root:
                # 1. 处理 QPOS (Observation)
                full_qpos = root['/observations/joint_pos'][()]
                # 左臂关节索引 (0,2,4...)
                q_left_index = [0,2,4,6,8,10,12]
                # 切片: 只取左臂7个关节
                qpos = full_qpos[:, q_left_index]

                # 2. 处理 Action (动作)
                # 关节部分
                action_joints_full = root['/action/joint_positions'][()]
                action_joints = action_joints_full[:, q_left_index]
                
                # 夹爪部分 (你说只用了1个自由度，所以取第0个是正确的)
                action_gripper_full = root['/action/gripper_command'][()]
                action_gripper = action_gripper_full[:, 0:1] 
                
                # 合并: 7 + 1 = 8 维
                action = np.concatenate([action_joints, action_gripper], axis=-1)
                
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
        except Exception as e:
            print(f"读取文件 {file_path} 出错: {e}")

    # 堆叠所有数据
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # 计算均值和方差
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    return {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze()
    }

# === 主程序 ===
if __name__ == "__main__":
    try:
        # 1. 查找文件
        files = find_all_hdf5(DATASET_DIR)
        
        # 2. 计算
        stats = get_norm_stats(files, NUM_EPISODES_TO_PROCESS)
        
        # 3. 打印结果
        def fmt(arr):
            # 格式化输出，方便直接复制
            return "np.array([" + ", ".join([f"{x:.5f}" for x in arr]) + "])"

        print("\n" + "="*20 + " 复制以下内容到 constants.py " + "="*20)
        print(f"'action_mean': {fmt(stats['action_mean'])},")
        print(f"'action_std': {fmt(stats['action_std'])},")
        print(f"'qpos_mean': {fmt(stats['qpos_mean'])},")
        print(f"'qpos_std': {fmt(stats['qpos_std'])},")
        print("="*60)
        
        # 验证维度
        print(f"\n[验证] 维度检查 (应为 8):")
        print(f"Action Mean Dim: {stats['action_mean'].shape[0]}")
        print(f"Qpos Mean Dim:   {stats['qpos_mean'].shape[0]}")
        
    except Exception as e:
        print(f"\n运行出错: {e}")