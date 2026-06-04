import json
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from ppo import PPO


def make_cartpole_env():
    return gym.make('CartPole-v1')


def create_vector_env(num_envs, vector_env_mode):
    env_fns = [make_cartpole_env for _ in range(num_envs)]
    if vector_env_mode == 'async':
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)


def resolve_device(device_name):
    if device_name in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        return 'cpu'
    return device_name


def run_training(config, seed, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    config = dict(config)
    config['device'] = resolve_device(config.get('device'))
    config.setdefault('num_envs', 32)
    config.setdefault('vector_env_mode', 'sync')
    config.setdefault('mini_batch_size', 1024)
    config.setdefault('torch_num_threads', 1)

    device = config['device']
    num_envs = config['num_envs']
    t_horizon = config['T_horizon']
    episodes = config.get('episodes', 500)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    torch.set_num_threads(max(1, int(config['torch_num_threads'])))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    env = create_vector_env(num_envs, config['vector_env_mode'])
    obs, _ = env.reset(seed=seed)

    model = PPO(
        lr=config['learning_rate'],
        gamma=config['gamma'],
        lmbda=config['lmbda'],
        eps_clip=config['eps_clip'],
        K_epoch=config['K_epoch'],
        entropy_coef=config['entropy_coef'],
        value_coef=config['value_coef'],
        grad_clip_norm=config['grad_clip_norm'],
        use_adv_norm=config['use_adv_norm'],
        use_entropy=config['use_entropy'],
        use_grad_clip=config['use_grad_clip'],
        mini_batch_size=config['mini_batch_size'],
        device=device,
    )

    episode_logs = []
    train_metrics_logs = []
    returns = []
    current_returns = np.zeros(num_envs, dtype=np.float32)
    completed_episodes = 0
    update_idx = 0
    time_to_solve = -1

    while completed_episodes < episodes:
        rollout = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
        }
        pending_episode_logs = []

        for _ in range(t_horizon):
            state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device)

            with torch.no_grad():
                prob = model.pi(state_tensor, softmax_dim=1)
                dist = torch.distributions.Categorical(prob)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            next_obs, rewards, terminated, truncated, _ = env.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)

            rollout['states'].append(state_tensor)
            rollout['actions'].append(actions)
            rollout['rewards'].append(
                torch.as_tensor(rewards / 100.0, dtype=torch.float32, device=model.device)
            )
            rollout['next_states'].append(
                torch.as_tensor(next_obs, dtype=torch.float32, device=model.device)
            )
            rollout['dones'].append(
                torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=model.device)
            )
            rollout['log_probs'].append(log_probs)

            current_returns += rewards
            for env_idx in np.flatnonzero(dones):
                if completed_episodes < episodes:
                    episode_return = float(current_returns[env_idx])
                    returns.append(episode_return)
                    ma_20 = float(np.mean(returns[-20:]))
                    ma_50 = float(np.mean(returns[-50:]))

                    if time_to_solve == -1 and ma_50 >= 475.0:
                        time_to_solve = completed_episodes

                    pending_episode_logs.append({
                        'episode': completed_episodes,
                        'return': episode_return,
                        'ma_20': ma_20,
                        'ma_50': ma_50,
                    })
                    completed_episodes += 1

                current_returns[env_idx] = 0.0

            obs = next_obs
            if completed_episodes >= episodes:
                break

        rollout = {key: torch.stack(value) for key, value in rollout.items()}
        metrics = model.train_net(rollout)

        if metrics:
            train_metrics_entry = dict(metrics)
            train_metrics_entry['episode'] = completed_episodes - 1 if completed_episodes > 0 else 0
            train_metrics_entry['update'] = update_idx
            train_metrics_logs.append(train_metrics_entry)

            for log_entry in pending_episode_logs:
                enriched_entry = dict(log_entry)
                enriched_entry.update(metrics)
                episode_logs.append(enriched_entry)
        else:
            episode_logs.extend(pending_episode_logs)

        update_idx += 1

    env.close()

    pd.DataFrame(episode_logs).to_csv(os.path.join(run_dir, 'episode_logs.csv'), index=False)
    if train_metrics_logs:
        pd.DataFrame(train_metrics_logs).to_csv(os.path.join(run_dir, 'train_metrics.csv'), index=False)

    final_100_returns = returns[-100:] if len(returns) >= 100 else returns
    summary = {
        'seed': seed,
        'num_envs': num_envs,
        'updates': update_idx,
        'final_100_mean': float(np.mean(final_100_returns)),
        'final_100_std': float(np.std(final_100_returns)),
        'time_to_solve': time_to_solve,
    }

    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, 'summary.csv'), index=False)
    return summary
