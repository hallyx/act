import torch
import numpy as np
import os
import pickle
import argparse
import zmq
from einops import rearrange
from copy import deepcopy

from constants import SIM_TASK_CONFIGS
from policy import ACTPolicy
from imitate_episodes import make_policy

# === 配置区域 ===
PORT = "5555"          # ZMQ 端口
CKPT_DIR = './ckpt' # 你的模型路径
TASK_NAME = 'astrobench_dual_arm'   # 你的任务名
# ================

class ACTInferenceServer:
    def __init__(self, ckpt_dir, task_name):
        self.ckpt_dir = ckpt_dir
        self.task_name = task_name
        
        # 1. 加载配置和统计数据
        self.task_config = SIM_TASK_CONFIGS[task_name]
        self.camera_names = self.task_config['camera_names']
        
        # 加载归一化统计数据
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
            
        # 2. 初始化模型
        # 硬编码模型参数 (必须和你训练时一致!)
        policy_config = {
            'lr': 1e-5, 'num_queries': 50, 'kl_weight': 10, 'hidden_dim': 512, 'dim_feedforward': 3200,
            'lr_backbone': 1e-5, 'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
            'camera_names': self.camera_names,
            'state_dim': 8  # <--- 你的 8 维状态
        }
        
        self.policy = make_policy('ACT', policy_config)
        # 加载权重
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        state_dict = torch.load(ckpt_path)
        self.policy.load_state_dict(state_dict)
        self.policy.cuda()
        self.policy.eval()
        print(f"Loaded model from {ckpt_path}")

        # 3. 初始化时间聚合 (Temporal Aggregation) 缓冲区
        self.chunk_size = policy_config['num_queries'] # 50
        self.state_dim = policy_config['state_dim']    # 8
        self.max_timesteps = 1000 # 预设一个最大步数，用于buffer
        
        # 这里的逻辑参考 eval_bc
        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.chunk_size, self.state_dim]).cuda()
        self.t = 0 # 当前时间步

    def pre_process(self, qpos_numpy):
        # 归一化 qpos
        return (qpos_numpy - self.stats['qpos_mean']) / self.stats['qpos_std']

    def post_process(self, action_numpy):
        # 反归一化 action
        return action_numpy * self.stats['action_std'] + self.stats['action_mean']

    def predict(self, qpos, images):
        """
        qpos: (8,) numpy array
        images: dict of (480, 640, 3) numpy arrays
        """
        with torch.inference_mode():
            # === 数据预处理 ===
            # 1. 处理 QPOS
            qpos_norm = self.pre_process(qpos)
            qpos_tensor = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0) # (1, 8)
            
            # 2. 处理图像
            curr_images = []
            for cam_name in self.camera_names:
                # images[cam_name] 应该是 RGB (H, W, C)
                curr_img = rearrange(images[cam_name], 'h w c -> c h w')
                curr_images.append(curr_img)
            curr_image_stack = np.stack(curr_images, axis=0)
            # 归一化 / 255.0
            image_tensor = torch.from_numpy(curr_image_stack / 255.0).float().cuda().unsqueeze(0) # (1, num_cam, C, H, W)

            # === 模型推理 (每 chunk_size 步推理一次，或者每步都推理) ===
            # 为了最平滑的效果，我们使用 Temporal Aggregation，每步都推理
            all_actions = self.policy(qpos_tensor, image_tensor) # (1, chunk_size, 8)
            
            # === 时间聚合 (核心逻辑) ===
            # 将当前预测的50步动作，填充到大表中
            self.all_time_actions[[self.t], self.t : self.t + self.chunk_size] = all_actions
            
            # 获取当前时间步 t 的所有重叠预测
            actions_for_curr_step = self.all_time_actions[:, self.t]
            
            # 找出非零的预测 (有效的预测)
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            
            # 加权平均 (指数加权，越新的预测权重越大)
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            
            # === 后处理 ===
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            
            self.t += 1 # 时间步前进
            
            return action

    def reset(self):
        self.t = 0
        self.all_time_actions.fill_(0)
        print("Model buffer reset.")

def main():
    # ZMQ Setup
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"ACT Server listening on port {PORT}...")

    # Inference Engine
    engine = ACTInferenceServer(CKPT_DIR, TASK_NAME)

    while True:
        # 1. 接收数据 (Pickle 反序列化)
        message = socket.recv()
        try:
            data = pickle.loads(message)
            
            # 简单指令处理
            if data == 'RESET':
                engine.reset()
                socket.send(pickle.dumps('OK'))
                continue
            
            # 2. 提取数据
            qpos = data['qpos']   # Expecting (8,) numpy
            images = data['images'] # Expecting dict of numpy
            
            # 3. 推理
            action = engine.predict(qpos, images)
            
            # 4. 发送动作
            socket.send(pickle.dumps(action))
            # print(f"Step {engine.t}: Action sent.")

        except Exception as e:
            print(f"Error: {e}")
            socket.send(pickle.dumps(None))

if __name__ == '__main__':
    main()