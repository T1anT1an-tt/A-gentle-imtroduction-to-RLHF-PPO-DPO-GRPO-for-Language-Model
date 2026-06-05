# PPO 连续动作空间实现解析

离散动作 PPO 和连续动作 PPO 的核心优化目标并没有改变。两者的主要区别在于：策略网络输出的内容不同，动作概率的解释方式也不同。

在离散动作空间中，actor 通常输出每个动作的概率；在连续动作空间中，actor 通常输出一个连续概率分布的参数，例如正态分布的均值和标准差。随后，动作从该连续分布中采样得到。

## 1. 神经网络本身并不知道动作空间是连续的

神经网络本质上只是一个参数化函数：

$$
f_\theta(s) \rightarrow \text{一组数值}
$$

这些数值具体表示什么，并不是网络自己决定的，而是由代码中的建模方式决定的。

在离散 PPO 中，网络输出会被解释为动作类别的概率。例如：

```python
self.fc_pi = nn.Linear(256, 2)

x = self.fc_pi(x)
prob = F.softmax(x, dim=softmax_dim)
```

这表示 actor 将状态 $s$ 映射为两个动作的概率：

$$
s \rightarrow [p_0, p_1]
$$

随后代码使用类别分布采样动作：

```python
m = Categorical(prob)
a = m.sample().item()
```

因此，离散策略可以写为：

$$
a \sim \operatorname{Categorical}(p_0, p_1)
$$

连续动作 PPO 的处理方式不同。网络不再输出每个离散动作的概率，而是输出连续概率分布的参数。

## 2. 连续 PPO 通过分布参数表示策略

连续动作空间中，动作可以是任意实数或某个区间内的实数。例如 `Pendulum-v1` 的动作范围近似为：

$$
a \in [-2, 2]
$$

这类动作不能像离散动作一样枚举所有可能值。常见做法是让 actor 输出正态分布的两个参数：

$$
\mu_\theta(s), \sigma_\theta(s)
$$

代码中对应两个输出头：

```python
self.fc_mu = nn.Linear(128, 1)
self.fc_std = nn.Linear(128, 1)
```

前向传播时：

```python
mu = 2.0 * torch.tanh(self.fc_mu(x))
std = F.softplus(self.fc_std(x))
return mu, std
```

这两个输出的含义是：

```text
mu  = 当前状态下动作分布的均值
std = 当前状态下动作分布的标准差
```

随后策略被定义为正态分布：

```python
dist = Normal(mu, std)
a = dist.sample()
```

数学上可以写为：

$$
a \sim \mathcal{N}\bigl(\mu_\theta(s), \sigma_\theta(s)\bigr)
$$

因此，连续 PPO 不是让神经网络自动判断“这是连续动作任务”，而是人为规定 actor 输出连续概率分布的参数，并从该分布中采样动作。

## 3. 网络学习的是从状态到分布参数的映射

连续 PPO 中，actor 学习的映射是：

$$
s \rightarrow \mu_\theta(s), \sigma_\theta(s)
$$

也就是说，给定当前状态 $s$，网络需要学会：

- 输出合适的动作均值 $\mu$，表示当前更倾向于选择哪个动作区域。
- 输出合适的标准差 $\sigma$，表示探索范围有多大。

例如在某个 Pendulum 状态下，actor 可能输出：

```text
mu = 1.2
std = 0.3
```

这表示策略倾向于输出 1.2 附近的动作，同时允许在该区域附近进行随机探索。

如果某些采样动作带来更高 advantage，PPO 更新会提高这些动作附近的概率密度；如果某些动作带来较低 advantage，PPO 更新会降低这些动作附近的概率密度。

## 4. PPO 如何更新连续动作策略

PPO 的关键项仍然是策略比值：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

连续动作版本通常使用 log probability 计算该比值：

```python
log_prob = dist.log_prob(a)
ratio = torch.exp(log_prob - old_log_prob)
```

其中：

```text
log_prob     = 新策略下动作 a 的 log probability
old_log_prob = 旧策略下动作 a 的 log probability
```

因此：

$$
r_t(\theta)
=
\exp\left(
\log \pi_\theta(a_t \mid s_t)
-
\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)
\right)
$$

如果该动作的 advantage 为正，说明它比当前价值估计预期更好，PPO 会提高该动作附近的概率密度。反之，如果 advantage 为负，PPO 会降低该动作附近的概率密度。

连续 PPO 的整体训练流程可以概括为：

```text
状态 s 输入 actor
actor 输出 mu, std
从 Normal(mu, std) 中采样动作 a
环境返回 reward 和 next state
计算 TD target 与 advantage
用 PPO clipped objective 更新策略
```

这就是“用神经网络拟合连续动作分布参数”的含义。

## 5. 分布形式由人类建模决定

连续动作策略的分布形式需要人为设定。常见选择包括：

- **Normal 分布**：最常见，适用于许多连续控制任务。
- **Tanh-Normal 分布**：适合动作有明确上下界的任务。
- **Beta 分布**：天然定义在有限区间内。
- **Deterministic policy**：DDPG、TD3 等算法常用。

对于本文代码，actor 的建模方式由三部分共同决定：

第一，定义两个输出头：

```python
self.fc_mu
self.fc_std
```

第二，将两个输出解释为正态分布的均值和标准差：

```python
mu = 2.0 * torch.tanh(self.fc_mu(x))
std = F.softplus(self.fc_std(x))
```

第三，从该正态分布中采样动作：

```python
dist = Normal(mu, std)
a = dist.sample()
```

然后将动作送入环境：

```python
env.step([a.item()])
```

因此，连续性来自代码中的策略分布设定，而不是神经网络自身的语义理解。

## 6. PPO 主公式不变

无论是离散动作还是连续动作，PPO 都在优化 clipped surrogate objective：

$$
L^{\operatorname{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}
\left(
r_t(\theta), 1-\epsilon, 1+\epsilon
\right) A_t
\right)
\right]
$$

其中最核心的是：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

也就是：

```text
新策略下采取该动作的可能性
/
旧策略下采取该动作的可能性
```

离散 PPO 中，代码通常直接使用动作概率：

```python
ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
```

连续 PPO 中，代码使用 log probability：

```python
ratio = torch.exp(log_prob - old_log_prob)
```

两者本质相同，区别只在于 $\pi_\theta(a \mid s)$ 的表示方式不同。

## 7. 离散动作：策略输出每个动作的概率

以 `CartPole-v1` 为例，动作空间是离散的：

$$
a \in \{0, 1\}
$$

actor 输出两个动作的概率：

$$
\pi_\theta(s) = [p_0, p_1]
$$

例如：

$$
\pi_\theta(s) = [0.3, 0.7]
$$

表示动作 0 的概率为 0.3，动作 1 的概率为 0.7。

代码结构为：

```python
self.fc_pi = nn.Linear(256, 2)
prob = F.softmax(x, dim=softmax_dim)
m = Categorical(prob)
a = m.sample().item()
```

因此，离散 actor 的策略形式是：

$$
a \sim \operatorname{Categorical}(\pi_\theta(s))
$$

## 8. 连续动作：策略输出概率分布参数

以 `Pendulum-v1` 为例，动作空间是连续区间：

$$
a \in [-2, 2]
$$

动作可以是：

```text
-1.72
0.35
1.89
-0.04
```

由于连续动作有无限多个取值，actor 不能为每一个具体动作单独输出概率。因此，连续 PPO 通常让 actor 输出正态分布参数：

$$
\pi_\theta(a \mid s)
=
\mathcal{N}\bigl(\mu_\theta(s), \sigma_\theta(s)\bigr)
$$

代码对应为：

```python
self.fc_mu = nn.Linear(128, 1)
self.fc_std = nn.Linear(128, 1)

mu = 2.0 * torch.tanh(self.fc_mu(x))
std = F.softplus(self.fc_std(x))
return mu, std
```

其中：

```text
mu  = 动作均值，表示策略倾向的动作中心
std = 动作标准差，表示探索范围
```

## 9. 连续动作中使用的是概率密度

离散动作中，$\pi_\theta(a \mid s)$ 是真实概率。例如：

$$
\pi_\theta(a=1 \mid s) = 0.7
$$

表示选择动作 1 的概率为 0.7。

连续动作中，严格来说，某个具体动作值的概率为 0：

$$
P(a = 0.35812) = 0
$$

因此，连续策略中计算的是动作在当前分布下的概率密度，而不是离散意义上的点概率。

这也是连续 PPO 通常存储 `log_prob` 的原因：

```python
dist = Normal(mu, std)
a = dist.sample()
log_prob = dist.log_prob(a)
```

采样时记录旧策略的 log probability：

```python
rollout.append((s, a, r / 10.0, s_prime, log_prob.item(), done))
```

训练时重新计算新策略下的 log probability：

```python
mu, std = self.pi(s, softmax_dim=1)
dist = Normal(mu, std)
log_prob = dist.log_prob(a)
ratio = torch.exp(log_prob - old_log_prob)
```

这与离散 PPO 中计算新旧策略概率比的逻辑完全一致。

## 10. 离散 PPO 与连续 PPO 的横向对比

| 部分 | 离散 PPO | 连续 PPO |
| --- | --- | --- |
| 动作空间 | $a \in \{0,1,2,\dots\}$ | $a \in \mathbb{R}$ 或有限连续区间 |
| policy 形式 | Categorical 分布 | Normal 分布等连续分布 |
| actor 输出 | 每个动作的概率 | 动作分布的均值 $\mu$ 和标准差 $\sigma$ |
| 采样方式 | $a \sim \operatorname{Categorical}(p)$ | $a \sim \mathcal{N}(\mu,\sigma)$ |
| PPO clip | 不变 | 不变 |
| advantage | 不变 | 不变 |
| critic | 不变 | 不变 |

## 11. 关键对应关系

离散版本：

```python
prob = model.pi(s)
m = Categorical(prob)
a = m.sample()
old_prob = prob[a]
```

数学上表示为：

$$
a \sim \operatorname{Categorical}(\pi_\theta(s))
$$

$$
old\_prob = \pi_{\theta_{\text{old}}}(a \mid s)
$$

连续版本：

```python
mu, std = model.pi(s)
dist = Normal(mu, std)
a = dist.sample()
old_log_prob = dist.log_prob(a)
```

数学上表示为：

$$
a \sim \mathcal{N}\bigl(\mu_\theta(s), \sigma_\theta(s)\bigr)
$$

$$
old\_log\_prob
=
\log \pi_{\theta_{\text{old}}}(a \mid s)
$$

## 总结

PPO 的优化目标在离散动作和连续动作中保持一致。变化的是策略分布的表示方式：

```text
离散动作：
actor 直接输出每个动作的概率，
通过 Categorical 分布采样动作。

连续动作：
actor 输出连续分布的参数，
通过 Normal(mu, std).log_prob(a) 计算动作密度。
```

因此，连续 PPO 的核心改动不是替换 PPO 算法本身，而是把策略模型从“输出动作类别概率”改成“输出连续动作分布参数”。
