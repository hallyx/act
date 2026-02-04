#!/usr/bin/env python3
import os

# === GPU 设置 ===
# 限制可见 GPU，避免与 Isaac Sim 冲突（如果在同一台机器上运行）
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 如果是独立显卡运行，请根据实际情况修改

import torch
import numpy as np
import pickle
import zmq
from einops import rearrange
from copy import deepcopy

# 从你的项目结构中导入 ACTPolicy
# 假设你的目录结构是:
# project/
#   act_server.py
#   policy.py (ACT模型定义)
#   detr/ ...
from policy import ACTPolicy

# === 配置区域 ===
PORT = "5555"
CKPT_DIR = './ckpt_ik' # 你的模型权重路径
TASK_NAME = 'astrobench_dual_arm'

# 相机配置（必须和客户端 run_client.py 中的 key 匹配！）
CAMERA_NAMES = ['rgb_main', 'rgb_left', 'rgb_right', 'rgb_under']
# ================

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

class ACTInferenceServer:
    def __init__(self, ckpt_dir, task_name, camera_names):
        self.ckpt_dir = ckpt_dir
        self.task_name = task_name
        self.camera_names = camera_names
        
        print(f"[Init] Task: {task_name}")
        print(f"[Init] Camera names: {camera_names}")
        
        # 1. 加载归一化统计数据
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        print("[Init] Loaded dataset stats.")
            
        # 2. 初始化模型配置
        # [关键] 必须与训练时的参数完全一致！
        # State Dim = 24:
        #   Base Pos (3) + Base Quat(4) ? (看你训练数据, 假设是7) + Joint Pos(14) ? 
        #   根据你的 Isaac Sim 发送的数据:
        #   qpos_16d = [Base Pos(3), Base Vel(6), Left Arm(7), Left Grip(1)] = 17 维
        #   请确认你的训练数据到底是 17 还是 24?
        #   根据你之前的描述 action_dim=16 (7pos+1grip+7vel+1grip_vel)
        #   这里假设 state_dim 是 17 (和你原来的代码一致)
        policy_config = {
            'lr': 1e-5, 
            #'num_queries': 20, # chunk size
            #'kl_weight': 0.10, 
            'num_queries': 50,
            'kl_weight': 10,
            'hidden_dim': 512, 
            'dim_feedforward': 3200,
            'lr_backbone': 5e-6, 
            'backbone': 'resnet18', 
            'enc_layers': 4, 
            'dec_layers': 7, 
            'nheads': 8,
            'camera_names': self.camera_names,
            'state_dim': 17,  # [必须确认] 客户端发送的是 [BasePos(3) + BaseVel(6) + ArmPos(7) + Grip(1)]
            'action_dim': 16  # [必须确认] 输出动作 [ArmPos(7) + Grip(1) + ArmVel(7) + GripVel(1)]
        }
        
        self.policy = make_policy('ACT', policy_config)
        
        # 加载权重
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        state_dict = torch.load(ckpt_path)
        self.policy.load_state_dict(state_dict)
        self.policy.cuda()
        self.policy.eval()
        print(f"[Init] Loaded model from {ckpt_path}")

        # 3. 初始化时间聚合缓冲区
        self.chunk_size = policy_config['num_queries']
        self.state_dim = policy_config['state_dim']
        self.action_dim = policy_config['action_dim']
        self.max_timesteps = 2000 # 足够覆盖最长 episode
        
        self.all_time_actions = torch.zeros([
            self.max_timesteps,
            self.max_timesteps + self.chunk_size,
            self.action_dim
        ]).cuda()
        self.t = 0 

    def pre_process(self, qpos_numpy):
        # 归一化输入
        qpos_arr = np.asarray(qpos_numpy, dtype=np.float32)
        if qpos_arr.shape[-1] != self.state_dim:
            # 自动修复维度不匹配 (如果可能)
            # 例如客户端可能发了 24 维，但模型需要 17 维
            print(f"[Warning] Dim mismatch: Model expects {self.state_dim}, got {qpos_arr.shape[-1]}")
            # 如果你知道怎么切片，可以在这里切
            
        mean = self.stats['qpos_mean']
        std = self.stats['qpos_std']
        # 防止除零
        std = np.where(std < 1e-6, 1.0, std)
        return (qpos_arr - mean) / std

    def post_process(self, action_numpy):
        # 反归一化输出
        mean = self.stats['action_mean']
        std = self.stats['action_std']
        return action_numpy * std + mean

    def predict(self, qpos, images):
        with torch.inference_mode():
            # 1. 预处理
            qpos_norm = self.pre_process(qpos)
            qpos_tensor = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)
            
            curr_images = []
            for cam_name in self.camera_names:
                # 客户端发送的是 (H, W, C), 模型需要 (C, H, W)
                curr_img = rearrange(images[cam_name], 'h w c -> c h w')
                curr_images.append(curr_img)
            curr_image_stack = np.stack(curr_images, axis=0)
            
            # 归一化图像 [0, 255] -> [0, 1]
            image_tensor = torch.from_numpy(curr_image_stack / 255.0).float().cuda().unsqueeze(0)

            # 2. 模型推理 -> (1, chunk_size, action_dim)
            all_actions = self.policy(qpos_tensor, image_tensor)
            
            # 3. 时间聚合 (Temporal Aggregation)
            # 将当前预测的动作块填入缓冲区
            # 缓冲区结构：[Time_Step_Started, Time_Step_Effective, Action_Dim]
            self.all_time_actions[[self.t], self.t : self.t + self.chunk_size] = all_actions
            
            # 获取当前时刻 t 的所有有效动作预测（来自过去 k 次预测）
            actions_for_curr_step = self.all_time_actions[:, self.t]
            
            # 过滤掉还未发生的预测（全0行）
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            
            # 指数加权平均
            k = 0.1
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            
            # 4. 后处理
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            
            # 时间步自增
            self.t += 1
            
            return action

    def reset(self):
        """
        清空时间聚合缓冲区，重置时间步
        """
        self.t = 0
        self.all_time_actions.fill_(0)
        print("[Server] Temporal aggregation buffer reset.")

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"[Server] ACT Inference Server listening on port {PORT}...")
    
    # 初始化引擎
    engine = ACTInferenceServer(CKPT_DIR, TASK_NAME, CAMERA_NAMES)

    while True:
        try:
            # 1. 接收消息
            message = socket.recv()
            data = pickle.loads(message)
            
            # 2. 处理复位指令 (匹配客户端 run_client.py 的 send_reset_signal_to_server)
            # 客户端发送: {'command': 'reset_policy', ...}
            if isinstance(data, dict) and data.get('command') == 'reset_policy':
                print("[Server] Received RESET command.")
                engine.reset()
                socket.send(pickle.dumps({'status': 'RESET_OK'}))
                continue
                
            # 3. 处理普通推理请求
            # 客户端发送: {'qpos': ..., 'images': ..., 'command': 'step'}
            qpos = data['qpos']
            images = data['images']
            
            # 4. 执行推理
            action = engine.predict(qpos, images)
            
            # 5. 返回结果 (包含 16维 动作)
            response = {
                'joint_positions': action, # 这里实际上包含了 pos(8) + vel(8)
                'status': 'OK'
            }
            socket.send(pickle.dumps(response))
            
            # Log first step
            if engine.t == 1:
                print(f"[Server] Episode started. Output action shape: {action.shape}")
            
        except Exception as e:
            print(f"[Server Error] {e}")
            # 发送错误响应防止客户端卡死
            try:
                socket.send(pickle.dumps({'status': 'ERROR', 'message': str(e)}))
            except:
                pass

if __name__ == '__main__':
    main()