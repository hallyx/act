import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import gc
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data, print_timing_stats, log_timing # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    return distributed, rank, world_size, local_rank


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def build_ddp_dataloaders(train_dataloader, val_dataloader, batch_size_train, batch_size_val, num_workers, rank, world_size):
    train_dataset = train_dataloader.dataset
    val_dataset = val_dataloader.dataset

    train_workers = max(0, int(num_workers))
    # 验证阶段不需要并行加载，避免 train/val 两套 worker 同时占用内存导致 OOM
    val_workers = 0

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=batch_size_train,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=False,
        num_workers=train_workers,
        persistent_workers=False,
    )
    if train_workers > 0:
        # 降低预取深度，减少每个 worker 的队列缓存占用
        train_loader_kwargs["prefetch_factor"] = 1

    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=batch_size_val,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=False,
        num_workers=val_workers,
        persistent_workers=False,
    )

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)
    return train_loader, val_loader


def reduce_epoch_summary(epoch_summary, device, distributed):
    if not distributed:
        return epoch_summary

    world_size = dist.get_world_size()
    reduced = {}
    for key, value in epoch_summary.items():
        tensor = value.detach().to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / world_size
        reduced[key] = tensor.cpu()
    return reduced


def init_running_stats(keys):
    return {k: 0.0 for k in keys}


def update_running_stats(running_stats, forward_dict):
    for k, v in forward_dict.items():
        running_stats[k] += float(v.detach().item())


def finalize_running_stats(running_stats, count):
    if count == 0:
        return {k: torch.tensor(0.0) for k in running_stats.keys()}
    return {k: torch.tensor(v / count) for k, v in running_stats.items()}


@record
def main(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 指定使用 GPU 1,第二张卡，不需要时注释掉
    distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    num_obs = args.get('num_obs', 1)  # 获取观测窗口大小，默认为1
    loader_mode = args.get('loader_mode', 'full_episode')
    sliding_stride = args.get('sliding_stride', 1)
    dataloader_workers = args.get('dataloader_workers', 1)
    chunk_size = args['chunk_size'] if args.get('chunk_size') is not None else 50
    
    # get task parameters
    is_sim = 'True'
    from constants import SIM_TASK_CONFIGS
    
    # 不管任务名叫什么，都从本地 SIM_TASK_CONFIGS 里读
    task_config = SIM_TASK_CONFIGS[args['task_name']]
    #if is_sim:
    #    from constants import SIM_TASK_CONFIGS
    #    task_config = SIM_TASK_CONFIGS[task_name]
    #else:
    #    from aloha_scripts.constants import TASK_CONFIGS
    #    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 17  # 关节位置(8) + 基座位置(3) + 基座速度(6) = 17维
    action_dim = 16  # 关节位置(7) + 夹爪(1)+速度(7) + 夹爪速度(1) = 16维
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': chunk_size,
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'action_dim': action_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim, 'action_dim': action_dim}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    config.update({
        'distributed': distributed,
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'is_main': is_main,
    })

    if is_eval:
        if distributed and not is_main:
            cleanup_distributed(distributed)
            return
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        cleanup_distributed(distributed)
        return

    # Data Loading
    if args.get('mixed_data', False):
        from utils import load_mixed_data
        # 注意：这里 dataset_dir 通常是 './data'
        train_dataloader, val_dataloader, stats, _ = load_mixed_data(
            dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
            loader_mode=loader_mode, chunk_size=chunk_size, stride=sliding_stride, num_workers=dataloader_workers)
    else:
        train_dataloader, val_dataloader, stats, _ = load_data(
            dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
            loader_mode=loader_mode, chunk_size=chunk_size, stride=sliding_stride, num_workers=dataloader_workers)

    if distributed:
        train_dataloader, val_dataloader = build_ddp_dataloaders(
            train_dataloader, val_dataloader,
            batch_size_train=batch_size_train,
            batch_size_val=batch_size_val,
            num_workers=dataloader_workers,
            rank=rank,
            world_size=world_size,
        )

    # save dataset stats
    if is_main:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    if is_main:
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # save best checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

    cleanup_distributed(distributed)


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def count_parameters(model):
    # 计算所有 requires_grad=True 的参数（即参与训练的参数）
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    #total_params = count_parameters(policy)
    #print(f"Model Size: {total_params / 1e6:.2f}M parameters")
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    
    # === 修改：强制加 .float() 转换数据类型 ===
    image_data = image_data.cuda().float()
    qpos_data = qpos_data.cuda().float()   # <--- 关键修改：转为 float32
    action_data = action_data.cuda().float() # <--- 保险起见，这也加上
    is_pad = is_pad.cuda()

    # 处理新的数据格式：从 (batch, num_obs, num_cam, C, H, W) -> (batch, num_cam, C, H, W)
    #                  和 (batch, num_obs, state_dim) -> (batch, state_dim)
    # 当 num_obs=1 时，squeeze 掉 num_obs 维度以保持向后兼容性
    if image_data.ndim == 6:
        image_data = image_data.squeeze(1)  # (batch, num_cam, C, H, W)
    if qpos_data.ndim == 3:
        qpos_data = qpos_data.squeeze(1)    # (batch, state_dim)
        
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    distributed = config.get('distributed', False)
    local_rank = config.get('local_rank', 0)
    is_main = config.get('is_main', True)

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    optimizer = make_optimizer(policy_class, policy)
    policy.cuda()
    if distributed:
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    
    # 开启混合精度训练 (AMP)
    scaler = torch.cuda.amp.GradScaler()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    epoch_iter = tqdm(range(num_epochs)) if is_main else range(num_epochs)
    for epoch in epoch_iter:
        if distributed and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        if is_main:
            print(f'\n{"="*60}\nEpoch {epoch}\n{"="*60}')
        
        # validation
        val_start_time = time.time()
        if is_main:
            print(f"开始验证... (共 {len(val_dataloader)} 个batch)")
        with torch.no_grad():
            policy.eval()
            running_val_stats = None
            val_count = 0
            for batch_idx, data in enumerate(val_dataloader):
                t_forward = time.time()
                with torch.cuda.amp.autocast():
                    forward_dict = forward_pass(data, policy)
                log_timing("10_val_forward_pass", time.time() - t_forward)
                if running_val_stats is None:
                    running_val_stats = init_running_stats(forward_dict.keys())
                update_running_stats(running_val_stats, forward_dict)
                val_count += 1
                
                # 清理中间变量减少内存占用
                del data, forward_dict
                # if batch_idx % 200 == 0:  # 减少清理频率
                #     torch.cuda.empty_cache()
                # 显示进度
                if is_main and (batch_idx % 100 == 0 or batch_idx == len(val_dataloader) - 1):
                    elapsed = time.time() - val_start_time
                    progress = (batch_idx + 1) / len(val_dataloader) * 100
                    print(f"  验证进度: {batch_idx+1}/{len(val_dataloader)} ({progress:.1f}%) - 已耗时: {elapsed:.1f}s")
            epoch_summary = finalize_running_stats(running_val_stats, val_count)
            epoch_summary = reduce_epoch_summary(epoch_summary, device=torch.device("cuda", local_rank), distributed=distributed)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if is_main and epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                model_to_save = policy.module if hasattr(policy, "module") else policy
                best_ckpt_info = (epoch, min_val_loss, deepcopy(model_to_save.state_dict()))
            # 清理验证阶段的内存
            del running_val_stats
            torch.cuda.empty_cache()
        val_time = time.time() - val_start_time
        if is_main:
            print(f'Val loss:   {epoch_val_loss:.5f} (耗时: {val_time:.1f}s)')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # training
        train_start_time = time.time()
        policy.train()
        optimizer.zero_grad()
        
        running_train_stats = None
        train_count = 0
        
        for batch_idx, data in enumerate(train_dataloader):
            t_forward = time.time()
            with torch.cuda.amp.autocast():
                forward_dict = forward_pass(data, policy)
            log_timing("20_train_forward_pass", time.time() - t_forward)
            
            # backward
            t_backward = time.time()
            loss = forward_dict['loss']
            scaler.scale(loss).backward()
            log_timing("21_backward", time.time() - t_backward)
            
            t_optim = time.time()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            log_timing("22_optimizer_step", time.time() - t_optim)
            
            if running_train_stats is None:
                running_train_stats = init_running_stats(forward_dict.keys())
            update_running_stats(running_train_stats, forward_dict)
            train_count += 1

            # 清理中间变量
            del data, forward_dict, loss
            # 大幅减少 empty_cache 频率，这在 4090 上非常耗时，仅在内存压力极大时使用
            # if batch_idx % 500 == 0: 
            #     torch.cuda.empty_cache()
                
        epoch_summary = finalize_running_stats(running_train_stats, train_count)
        epoch_summary = reduce_epoch_summary(epoch_summary, device=torch.device("cuda", local_rank), distributed=distributed)
        epoch_train_loss = epoch_summary['loss']
        train_time = time.time() - train_start_time
        if is_main:
            print(f'Train loss: {epoch_train_loss:.5f} (耗时: {train_time:.1f}s)')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)
            train_history.append(epoch_summary)
        del running_train_stats
        
        # 每个epoch结束后强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        epoch_total_time = time.time() - epoch_start_time
        if is_main:
            print(f'Epoch {epoch} 总耗时: {epoch_total_time:.1f}s (验证:{val_time:.1f}s + 训练:{train_time:.1f}s)')
        
        # 每10个epoch打印一次详细统计并保存模型
        if is_main and epoch % 10 == 0:
            print_timing_stats()
            model_to_save = policy.module if hasattr(policy, "module") else policy
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(model_to_save.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    if not is_main:
        return None

    model_to_save = policy.module if hasattr(policy, "module") else policy
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(model_to_save.state_dict(), ckpt_path)

    if best_ckpt_info is None:
        best_ckpt_info = (num_epochs - 1, float('inf'), deepcopy(model_to_save.state_dict()))
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=float, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--num_obs', action='store', type=int, default=1, help='num_obs', required=False)
    parser.add_argument('--mixed_data', action='store_true',default=True, help='Use mixed data from multiple folders')
    parser.add_argument('--loader_mode', action='store', type=str, default='full_episode',
                        choices=['full_episode', 'sliding_window'],
                        help='Data loading mode for training samples')
    parser.add_argument('--sliding_stride', action='store', type=int, default=1,
                        help='Stride used when loader_mode=sliding_window')
    parser.add_argument('--dataloader_workers', action='store', type=int, default=1,
                        help='DataLoader num_workers')
    
    try:
        main(vars(parser.parse_args()))
    except Exception:
        import traceback
        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        print(f"[FATAL] rank={rank}, local_rank={local_rank}")
        traceback.print_exc()
        raise
