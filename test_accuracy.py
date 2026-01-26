import os

# === 必须在任何 torch 或 omni 模块导入之前设置！===
# 这将限制当前进程只看得到物理上的 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os
import torch
import numpy as np
import h5py
import pickle
from tqdm import tqdm
from copy import deepcopy

# Import model components (same as imitate_episodes.py)
from policy import ACTPolicy
from utils import get_norm_stats, load_data

# Configuration (matching the training setup)
CKPT_DIR = '/home/gpuserver/hx/github/act/ckpt'
DATA_DIR = '/home/gpuserver/hx/github/act/data'
TASK_NAME = 'astrobench_dual_arm'
CKPT_NAME = 'policy_best.ckpt' # or use specific epoch
STATE_DIM = 17
ACTION_DIM = 16
NUM_QUERIES = 50 # Updated from 100 to 50 based on checkpoint mismatch

# Gripper threshold for "success" judgment
# Based on constants.py PUPPET_GRIPPER_JOINT_CLOSE = -0.6213
GRIPPER_CLOSE_THRESHOLD = -0.05 # Adjust based on normalized values if needed 

def make_policy(policy_config):
    policy = ACTPolicy(policy_config)
    return policy

def load_policy(ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    
    # Identify state dict key or use full checkpoint if it matches state dict structure
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif any(k.startswith('model.') for k in checkpoint.keys()):
        # The keys themselves seem to be the state dict (flat structure)
        state_dict = checkpoint
    else:
        raise KeyError(f"Could not find model state dict in checkpoint. Keys: {list(checkpoint.keys())[:5]}...")

    # Use the same config as training
    policy_config = checkpoint.get('policy_config', {
        'lr': 1e-5,
        'num_queries': NUM_QUERIES,
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': ['rgb_main', 'rgb_left', 'rgb_under'],
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
    })
    
    policy = make_policy(policy_config)
    
    # Handle possible prefix mismatch (e.g., if saved with DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    policy.load_state_dict(new_state_dict)
    policy.cuda()
    policy.eval()
    print(f"Successfully loaded policy from {ckpt_path}")
    return policy, checkpoint.get('stats', None)

def eval_accuracy():
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    if not os.path.exists(ckpt_path):
        print(f"Ckpt not found: {ckpt_path}")
        return

    policy, stats = load_policy(ckpt_path)
    
    # If stats not in ckpt, we need to load or recalculate
    if stats is None:
        stats_path = os.path.join(CKPT_DIR, 'dataset_stats.pkl')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            print(f"Loaded stats from {stats_path}")
        else:
            print("Warning: Stats not found. Results will be incorrect.")
            # return

    # Find test files (e.g., exclude 'recovery' or use a specific subset)
    import glob
    all_files = glob.glob(os.path.join(DATA_DIR, '**/*.h5'), recursive=True)
    # Filter for expert files only for accuracy test
    test_files = [f for f in all_files if 'expert' in f]
    test_files.sort()
    
    # Take a subset for evaluation
    test_files = test_files[:20] 
    
    print(f"Evaluating on {len(test_files)} expert episodes...")

    mse_list = []
    traj_mse_list = []
    success_count = 0
    total_episodes = len(test_files)

    with torch.inference_mode():
        for file_path in tqdm(test_files):
            with h5py.File(file_path, 'r') as root:
                # 1. Determine Episode Length
                original_action_shape = root['/action/joint_positions'].shape
                episode_len = original_action_shape[0]
                
                # 2. Extract QPOS components (matching utils.py logic)
                # Left arm joint indices
                q_left_idx = [0, 2, 4, 6, 8, 10, 12]
                
                # 3. Ground Truth Actions (for MSE calculation)
                action_joints_pos = root['/action/joint_positions'][:, q_left_idx]
                action_gripper = root['/action/gripper_command'][:, 0:1] # Left gripper
                action_joints_vel = root['/action/joint_velocities'][:, q_left_idx]
                # Assuming gripper velocity is 0 if not present, as in utils.py
                try:
                    action_gripper_vel = root['/action/gripper_velocity'][:, 0:1]
                except:
                    action_gripper_vel = np.zeros_like(action_gripper)
                
                actions_gt = np.concatenate([action_joints_pos, action_gripper, 
                                            action_joints_vel, action_gripper_vel], axis=1) # [T, 16]

                episode_mse = []
                episode_traj_mse = []
                
                # Success Logic: Use the gripper angle from the expert's final state
                expert_final_gripper = actions_gt[-1, 7] # Position of gripper
                expert_is_success = expert_final_gripper < GRIPPER_CLOSE_THRESHOLD
                
                model_final_gripper = 0
                
                import cv2
                
                for t in range(episode_len):
                    # Prepare input QPOS (Matching utils.py)
                    base_pose = root['/observations/base_pose'][t]
                    base_pos = base_pose[:3]
                    base_vel = root['/observations/base_vel'][t]
                    full_qos = root['/observations/joint_pos'][t]
                    
                    # joint_qpos = concatenate(arm(7), gripper(1))
                    # Note: utils.py hardcoded gripper to 0.0, we should follow or use real if available
                    try:
                        qpos_gripper = root['/observations/gripper_pos'][t, 0:1]
                    except:
                        qpos_gripper = np.array([0.0])
                    
                    joint_qpos = np.concatenate([full_qos[q_left_idx], qpos_gripper]) # 8-dim
                    curr_qpos = np.concatenate([base_pos, base_vel, joint_qpos]) # 17-dim
                    
                    # Normalize QPOS
                    curr_qpos_norm = (curr_qpos - stats['qpos_mean']) / stats['qpos_std']
                    curr_qpos_torch = torch.from_numpy(curr_qpos_norm).float().cuda().unsqueeze(0)
                    
                    # Prepare Images (Matching utils.py decoding)
                    curr_images = []
                    for cam_name in ['rgb_main', 'rgb_left', 'rgb_under']:
                        # In utils.py: img_bytes = root[f'/observations/{cam_name}'][start_ts]
                        try:
                            img_bytes = root[f'/observations/{cam_name}'][t]
                        except KeyError:
                            # Try /observations/images/{cam_name}
                            img_bytes = root[f'/observations/images/{cam_name}'][t]

                        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        
                        img_torch = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
                        curr_images.append(img_torch)
                    
                    curr_images_torch = torch.stack(curr_images).cuda().unsqueeze(0) # [1, NumCam, 3, H, W]
                    
                    # Inference
                    all_actions = policy(curr_qpos_torch, curr_images_torch) # [1, NUM_QUERIES, ACTION_DIM]
                    
                    # Unnormalize ALL predicted actions in the chunk
                    all_actions_np = all_actions[0].cpu().numpy() # [NUM_QUERIES, ACTION_DIM]
                    all_actions_unnorm = all_actions_np * stats['action_std'] + stats['action_mean']
                    
                    # 1. Point MSE (First action only)
                    pred_action_unnorm = all_actions_unnorm[0]
                    gt_action = actions_gt[t]
                    mse = np.mean((pred_action_unnorm - gt_action)**2)
                    episode_mse.append(mse)
                    
                    # 2. Movement Magnitude Check
                    # Check if the model is predicting significant movement towards the goal
                    # Compare the first 10 steps of prediction vs ground truth
                    max_t_future = min(t + 10, episode_len)
                    future_len = max_t_future - t
                    if future_len > 0:
                        pred_future = all_actions_unnorm[:future_len, :7] # Arm joints only
                        gt_future = actions_gt[t:max_t_future, :7]
                        traj_mse = np.mean((pred_future - gt_future)**2)
                        episode_traj_mse.append(traj_mse)
                    
                    # Track gripper state at the end of the episode for success check
                    model_final_gripper = pred_action_unnorm[7]

                # Calculate metrics for the episode
                avg_mse = np.mean(episode_mse)
                avg_traj_mse = np.mean(episode_traj_mse) if episode_traj_mse else 0
                mse_list.append(avg_mse)
                traj_mse_list.append(avg_traj_mse)
                model_is_success = model_final_gripper < GRIPPER_CLOSE_THRESHOLD
                
                if expert_is_success:
                    if model_is_success:
                        success_count += 1
                else:
                    # If expert didn't "grasp", then it's a "success" if model also didn't grasp?
                    # Usually we only care about positive success cases
                    # For now, let's just count if it matches the expert's final intent
                    if not model_is_success:
                        success_count += 1

    print("\n--- Evaluation Results ---")
    print(f"Average Action MSE (Next Step): {np.mean(mse_list):.5f}")
    print(f"Average Trajectory MSE (Chunk): {np.mean(traj_mse_list):.5f}")
    print(f"Success Rate (Gripper Alignment): {success_count / total_episodes * 100:.2f}%")
    print(f"Total Episodes: {total_episodes}")

if __name__ == '__main__':
    eval_accuracy()

