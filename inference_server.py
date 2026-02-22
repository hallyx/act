#!/usr/bin/env python3
import os
import json
import argparse

import torch
import numpy as np
import pickle
import zmq
from einops import rearrange
from constants import SIM_TASK_CONFIGS

from policy import ACTPolicy


class NumpyCompatUnpickler(pickle.Unpickler):
    """Compat unpickler for numpy module path changes across versions."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        elif module.startswith("numpy.core"):
            module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


def load_pickle_compat(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            msg = str(e)
            if "numpy._core" in msg or "numpy.core" in msg:
                f.seek(0)
                print("[Init] Fallback to numpy-compatible unpickler for dataset stats.")
                return NumpyCompatUnpickler(f).load()
            raise

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    defaults = {
        "port": "5555",
        "cuda_visible_devices": "0",
        "ckpt_dir": "./ckpt/ckpt_ik_sliding/",
        "task_name": "astrobench_dual_arm",
        "camera_names": None,
        "max_timesteps": 2000,
        "temporal_agg_k": 0.1,
        "policy": {
            "policy_class": "ACT",
            "lr": 1e-5,
            "num_queries": 50,
            "kl_weight": 10,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "lr_backbone": 5e-6,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "state_dim": 17,
            "action_dim": 16
        }
    }

    merged = defaults.copy()
    merged.update(cfg)
    policy_cfg = defaults["policy"].copy()
    policy_cfg.update(cfg.get("policy", {}))
    merged["policy"] = policy_cfg

    if merged["camera_names"] is None:
        task_name = merged["task_name"]
        if task_name in SIM_TASK_CONFIGS:
            merged["camera_names"] = SIM_TASK_CONFIGS[task_name]["camera_names"]
        else:
            raise ValueError(
                f"camera_names not set in config and task '{task_name}' is not found in SIM_TASK_CONFIGS."
            )

    merged["policy"]["camera_names"] = merged["camera_names"]
    return merged

class ACTInferenceServer:
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = config["ckpt_dir"]
        self.task_name = config["task_name"]
        self.camera_names = config["camera_names"]
        self.temporal_agg_k = float(config["temporal_agg_k"])
        self.max_timesteps = int(config["max_timesteps"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[Init] Task: {self.task_name}")
        print(f"[Init] Camera names: {self.camera_names}")
        print(f"[Init] Device: {self.device}")
        
        # 1. 加载归一化统计数据
        stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
        self.stats = load_pickle_compat(stats_path)
        print("[Init] Loaded dataset stats.")
            
        # 2. 初始化模型配置（从配置文件读取）
        policy_config = config["policy"]
        
        self.policy = make_policy(policy_config.get("policy_class", "ACT"), policy_config)
        
        # 加载权重
        ckpt_path = os.path.join(self.ckpt_dir, 'policy_best.ckpt')
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()
        print(f"[Init] Loaded model from {ckpt_path}")

        # 3. 初始化时间聚合缓冲区
        self.chunk_size = policy_config['num_queries']
        self.state_dim = policy_config['state_dim']
        self.action_dim = policy_config['action_dim']
        
        self.all_time_actions = torch.zeros([
            self.max_timesteps,
            self.max_timesteps + self.chunk_size,
            self.action_dim
        ], dtype=torch.float32, device=self.device)
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
            qpos_tensor = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)
            
            curr_images = []
            for cam_name in self.camera_names:
                # 客户端发送的是 (H, W, C), 模型需要 (C, H, W)
                curr_img = rearrange(images[cam_name], 'h w c -> c h w')
                curr_images.append(curr_img)
            curr_image_stack = np.stack(curr_images, axis=0)
            
            # 归一化图像 [0, 255] -> [0, 1]
            image_tensor = torch.from_numpy(curr_image_stack / 255.0).float().to(self.device).unsqueeze(0)

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
            k = self.temporal_agg_k
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='inference_config.json', help='Path to inference JSON config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])
    port = str(cfg["port"])

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"[Server] ACT Inference Server listening on port {port}...")
    
    # 初始化引擎
    engine = ACTInferenceServer(cfg)

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
            episode_id = data.get('episode_id')
            step_id = data.get('step_id')
            timestamp = data.get('timestamp')
            
            # 4. 执行推理
            action = engine.predict(qpos, images)
            
            # 5. 返回结果 (包含 16维 动作)
            response = {
                'joint_positions': action, # 这里实际上包含了 pos(8) + vel(8)
                'status': 'OK'
            }
            if episode_id is not None:
                response['episode_id'] = episode_id
            if step_id is not None:
                response['step_id'] = step_id
            if timestamp is not None:
                response['timestamp'] = timestamp
            socket.send(pickle.dumps(response))
            
            # Log first step
            if engine.t == 1:
                print(f"[Server] Episode started. Output action shape: {action.shape}")
            
        except Exception as e:
            print(f"[Server Error] {e}")
            # 发送错误响应防止客户端卡死
            try:
                error_response = {'status': 'ERROR', 'message': str(e)}
                if isinstance(data, dict):
                    if data.get('episode_id') is not None:
                        error_response['episode_id'] = data.get('episode_id')
                    if data.get('step_id') is not None:
                        error_response['step_id'] = data.get('step_id')
                    if data.get('timestamp') is not None:
                        error_response['timestamp'] = data.get('timestamp')
                socket.send(pickle.dumps(error_response))
            except:
                pass

if __name__ == '__main__':
    main()
