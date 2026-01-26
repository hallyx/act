#!/usr/bin/env python3
"""
验证qpos维度和数据加载
"""
import h5py
import numpy as np
import os

def test_data_dimensions():
    """测试数据文件中的维度"""
    
    # 查找一个示例数据文件
    data_dir = '/home/gpuserver/hx/github/act/data'
    
    # 查找第一个recovery文件
    sample_file = None
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if 'recovery' in f and f.endswith('.h5'):
                sample_file = os.path.join(root, f)
                break
        if sample_file:
            break
    
    if not sample_file:
        print("❌ 未找到数据文件！")
        return False
    
    print(f"📁 测试文件: {sample_file}\n")
    
    try:
        with h5py.File(sample_file, 'r') as root:
            print("=== 数据集结构 ===")
            
            # 检查基座数据
            if '/observations/base_pose' in root:
                base_pose = root['/observations/base_pose'][()]
                print(f"✅ base_pose shape: {base_pose.shape} (期望: (T, 7))")
            else:
                print("❌ 缺少 /observations/base_pose")
                return False
            
            if '/observations/base_vel' in root:
                base_vel = root['/observations/base_vel'][()]
                print(f"✅ base_vel shape: {base_vel.shape} (期望: (T, 6))")
            else:
                print("❌ 缺少 /observations/base_vel")
                return False
            
            # 检查关节数据
            if '/observations/joint_pos' in root:
                joint_pos = root['/observations/joint_pos'][()]
                print(f"✅ joint_pos shape: {joint_pos.shape} (期望: (T, 28))")
            else:
                print("❌ 缺少 /observations/joint_pos")
                return False
            
            # 检查动作数据
            if '/action/joint_positions' in root:
                action_joints = root['/action/joint_positions'][()]
                print(f"✅ action joint_positions shape: {action_joints.shape}")
            
            if '/action/gripper_command' in root:
                action_gripper = root['/action/gripper_command'][()]
                print(f"✅ action gripper_command shape: {action_gripper.shape}")
            
            # 测试qpos构造
            print("\n=== 测试 qpos 构造 ===")
            start_ts = 0
            
            # 基座位置和速度
            base_pose_t = root['/observations/base_pose'][start_ts]
            base_pos = base_pose_t[:3]
            base_vel_t = root['/observations/base_vel'][start_ts]
            
            # 关节位置
            full_qpos = root['/observations/joint_pos'][start_ts]
            q_left_index = [0, 2, 4, 6, 8, 10, 12]
            qpos_gripper = np.array([0.0])
            joint_qpos = np.concatenate([full_qpos[q_left_index], qpos_gripper])
            
            # 组合qpos
            qpos = np.concatenate([base_pos, base_vel_t, joint_qpos])
            
            print(f"  base_pos: {base_pos.shape} = {base_pos}")
            print(f"  base_vel: {base_vel_t.shape} = {base_vel_t}")
            print(f"  joint_qpos: {joint_qpos.shape} = {joint_qpos}")
            print(f"  ✅ 最终 qpos shape: {qpos.shape} (期望: (17,))")
            
            if qpos.shape[0] == 17:
                print("\n✅ 所有维度检查通过！")
                return True
            else:
                print(f"\n❌ qpos维度错误！期望17，实际{qpos.shape[0]}")
                return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils_loading():
    """测试utils.py中的数据加载"""
    print("\n=== 测试 utils.py 数据加载 ===")
    
    try:
        from utils import find_all_hdf5, get_norm_stats
        
        data_dir = '/home/gpuserver/hx/github/act/data'
        
        # 查找文件
        print("查找数据文件...")
        file_paths = find_all_hdf5(data_dir)
        
        if len(file_paths) == 0:
            print("❌ 未找到数据文件")
            return False
        
        print(f"✅ 找到 {len(file_paths)} 个文件")
        
        # 测试计算统计信息
        print("\n计算归一化统计信息...")
        stats = get_norm_stats(file_paths, num_episodes=2)  # 只用2个episode测试
        
        print(f"✅ qpos_mean shape: {stats['qpos_mean'].shape} (期望: (17,))")
        print(f"✅ qpos_std shape: {stats['qpos_std'].shape}")
        print(f"✅ action_mean shape: {stats['action_mean'].shape} (期望: (16,))")
        print(f"✅ action_std shape: {stats['action_std'].shape}")
        
        if stats['qpos_mean'].shape[0] == 17 and stats['action_mean'].shape[0] == 16:
            print("\n✅ utils.py 数据加载测试通过！")
            return True
        else:
            print(f"\n❌ 维度错误！")
            print(f"   qpos期望17，实际{stats['qpos_mean'].shape[0]}")
            print(f"   action期望16，实际{stats['action_mean'].shape[0]}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("  维度验证工具")
    print("  State (qpos): base_pos(3) + base_vel(6) + joint_pos(8) = 17")
    print("  Action: joint_pos(7) + gripper(1) + joint_vel(7) + gripper_vel(1) = 16")
    print("=" * 60)
    
    success1 = test_data_dimensions()
    success2 = test_utils_loading()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("  ✅ 所有测试通过！")
    else:
        print("  ❌ 部分测试失败")
    print("=" * 60)
