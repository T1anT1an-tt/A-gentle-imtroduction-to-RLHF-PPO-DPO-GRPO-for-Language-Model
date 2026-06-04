import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PPO(nn.Module):
    def __init__(self, lr=0.0005, gamma=0.98, lmbda=0.95, eps_clip=0.1, K_epoch=3,
                 entropy_coef=0.01, value_coef=0.5, grad_clip_norm=0.5,
                 use_adv_norm=True, use_entropy=True, use_grad_clip=True,
                 mini_batch_size=None, device=None):
        super(PPO, self).__init__()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip_norm = grad_clip_norm
        self.use_adv_norm = use_adv_norm
        self.use_entropy = use_entropy
        self.use_grad_clip = use_grad_clip
        self.mini_batch_size = mini_batch_size

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def pi(self, x, softmax_dim=-1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        return F.softmax(x, dim=softmax_dim)

    def v(self, x):
        x = F.relu(self.fc1(x))
        return self.fc_v(x)

    def compute_gae(self, rewards, values, next_values, dones):
        masks = 1.0 - dones
        deltas = rewards + self.gamma * next_values * masks - values
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=self.device)

        for t in range(rewards.shape[0] - 1, -1, -1):
            gae = deltas[t] + self.gamma * self.lmbda * masks[t] * gae
            advantages[t] = gae

        td_target = advantages + values
        return td_target, advantages

    def train_net(self, rollout):
        if rollout is None or rollout['states'].numel() == 0:
            return None

        states = rollout['states'].to(self.device)
        actions = rollout['actions'].to(self.device)
        rewards = rollout['rewards'].to(self.device)
        next_states = rollout['next_states'].to(self.device)
        dones = rollout['dones'].to(self.device)
        old_log_probs = rollout['log_probs'].to(self.device)

        time_steps, num_envs = actions.shape
        obs_dim = states.shape[-1]

        with torch.no_grad():
            values = self.v(states.reshape(-1, obs_dim)).view(time_steps, num_envs)
            next_values = self.v(next_states.reshape(-1, obs_dim)).view(time_steps, num_envs)
            td_target, advantages = self.compute_gae(rewards, values, next_values, dones)

            if self.use_adv_norm and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        adv_mean = advantages.mean().item()
        adv_std = advantages.std(unbiased=False).item()

        flat_states = states.reshape(-1, obs_dim)
        flat_actions = actions.reshape(-1, 1)
        flat_old_log_probs = old_log_probs.reshape(-1, 1)
        flat_td_target = td_target.reshape(-1, 1)
        flat_advantages = advantages.reshape(-1, 1)

        batch_size = flat_states.shape[0]
        mini_batch_size = min(self.mini_batch_size or batch_size, batch_size)
        epoch_metrics = []

        for _ in range(self.K_epoch):
            permutation = torch.randperm(batch_size, device=self.device)
            batch_metrics = []

            for start in range(0, batch_size, mini_batch_size):
                idx = permutation[start:start + mini_batch_size]
                mb_states = flat_states[idx]
                mb_actions = flat_actions[idx]
                mb_old_log_probs = flat_old_log_probs[idx]
                mb_td_target = flat_td_target[idx]
                mb_advantages = flat_advantages[idx]

                pi = self.pi(mb_states, softmax_dim=1)
                dist = Categorical(pi)
                new_log_probs = dist.log_prob(mb_actions.squeeze(-1)).unsqueeze(-1)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.v(mb_states)
                critic_loss = F.smooth_l1_loss(values_pred, mb_td_target)
                entropy = dist.entropy().mean()

                total_loss = actor_loss + self.value_coef * critic_loss
                if self.use_entropy:
                    total_loss -= self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                with torch.no_grad():
                    ratio_val = ratio.detach()
                    clip_frac = (
                        (ratio_val < 1 - self.eps_clip) | (ratio_val > 1 + self.eps_clip)
                    ).float().mean().item()

                batch_metrics.append({
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'entropy': entropy.item(),
                    'total_loss': total_loss.item(),
                    'ratio_mean': ratio_val.mean().item(),
                    'ratio_std': ratio_val.std(unbiased=False).item(),
                    'ratio_min': ratio_val.min().item(),
                    'ratio_max': ratio_val.max().item(),
                    'clip_fraction': clip_frac,
                    'adv_mean': adv_mean,
                    'adv_std': adv_std,
                })

            epoch_metrics.append({
                key: np.mean([metric[key] for metric in batch_metrics])
                for key in batch_metrics[0]
            })

        return {
            key: np.mean([metric[key] for metric in epoch_metrics])
            for key in epoch_metrics[0]
        }
