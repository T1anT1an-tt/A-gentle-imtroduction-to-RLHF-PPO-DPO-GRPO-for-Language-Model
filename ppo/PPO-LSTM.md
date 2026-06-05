# PPO-LSTM 实现解析

PPO-LSTM 的主体训练逻辑仍然是 PPO。与普通 PPO 相比，LSTM 版主要在网络中加入了循环记忆层，并在采样、缓存和训练时额外维护 hidden state。

普通 PPO 每一步只根据当前状态选择动作；PPO-LSTM 会在当前状态之外引入历史记忆，因此更适合部分可观测环境或需要时间上下文的任务。

## 1. import 部分

| 位置 | 普通 PPO | PPO-LSTM |
| --- | --- | --- |
| import | 多了 `matplotlib.pyplot as plt` | 多了 `time`，但代码中未实际使用 |

普通 PPO 版本包含画图调试相关依赖，而 LSTM 版本保留了较接近 MinimalRL 原始代码的结构，没有额外加入训练曲线绘制模块。

## 2. 超参数对比

| 超参数 | 普通 PPO | PPO-LSTM | 含义 |
| --- | ---: | ---: | --- |
| `learning_rate` | 0.0005 | 0.0005 | 学习率相同 |
| `gamma` | 0.98 | 0.98 | 折扣因子相同 |
| `lmbda` | 0.95 | 0.95 | GAE 参数相同 |
| `eps_clip` | 0.1 | 0.1 | PPO clip 范围相同 |
| `K_epoch` | 3 | 2 | LSTM 版每批数据更新次数更少 |
| `T_horizon` | 20 | 20 | 每收集 20 步训练一次 |

整体来看，PPO 的核心超参数基本保持一致。LSTM 版将 `K_epoch` 从 3 调整为 2，可能是因为 LSTM 的反向传播更复杂，重复更新过多更容易带来不稳定。

## 3. `DebugTracker` 差异

普通 PPO 中包含调试记录模块：

```python
class DebugTracker:
    ...
```

该模块用于记录 episode、平均分，并在训练结束后绘制曲线。

PPO-LSTM 版本没有该模块，因此它更接近原始 MinimalRL 代码，还没有经过相同的实验日志与可视化重构。如果需要比较普通 PPO 与 PPO-LSTM 的稳定性，建议为 LSTM 版本补充同样的日志记录和绘图逻辑。

## 4. 网络结构差异

### 普通 PPO

```python
self.fc1   = nn.Linear(4, 256)
self.fc_pi = nn.Linear(256, 2)
self.fc_v  = nn.Linear(256, 1)
```

结构为：

```text
状态 s: 4 维
↓
全连接层 fc1: 4 -> 256
↓
策略头 fc_pi: 256 -> 2
价值头 fc_v: 256 -> 1
```

普通 PPO 每次只基于当前状态 $s$ 计算动作概率和状态价值，不显式保存历史信息。

### PPO-LSTM

```python
self.fc1   = nn.Linear(4, 64)
self.lstm  = nn.LSTM(64, 32)
self.fc_pi = nn.Linear(32, 2)
self.fc_v  = nn.Linear(32, 1)
```

结构变为：

```text
状态 s: 4 维
↓
全连接层 fc1: 4 -> 64
↓
LSTM: 64 -> 32
↓
策略头 fc_pi: 32 -> 2
价值头 fc_v: 32 -> 1
```

核心新增模块是：

```python
self.lstm = nn.LSTM(64, 32)
```

它使模型能够在当前状态之外利用前面时间步的隐藏记忆。对于 CartPole 这类状态信息较完整的环境，LSTM 不一定带来明显性能提升；但在当前观测不足以判断环境状态的任务中，LSTM 的历史记忆会更有价值。

## 5. `pi()` 策略函数

### 普通 PPO

```python
def pi(self, x, softmax_dim = 0):
    x = F.relu(self.fc1(x))
    x = self.fc_pi(x)
    prob = F.softmax(x, dim=softmax_dim)
    return prob
```

流程为：

```text
输入状态 x
↓
fc1
↓
fc_pi
↓
softmax 得到动作概率
```

返回值只有动作概率：

```python
prob
```

### PPO-LSTM

```python
def pi(self, x, hidden):
    x = F.relu(self.fc1(x))
    x = x.view(-1, 1, 64)
    x, lstm_hidden = self.lstm(x, hidden)
    x = self.fc_pi(x)
    prob = F.softmax(x, dim=2)
    return prob, lstm_hidden
```

LSTM 版多了两个关键对象：

```python
hidden
lstm_hidden
```

含义为：

```text
hidden      = 输入 LSTM 前的记忆
lstm_hidden = LSTM 处理当前状态后更新出的新记忆
```

因此调用方式从普通 PPO 的：

```python
prob = model.pi(s)
```

变为：

```python
prob, h_out = model.pi(s, h_in)
```

PPO-LSTM 的策略函数不仅输出动作概率，还会输出更新后的记忆状态。

## 6. `x.view(-1, 1, 64)` 的作用

PPO-LSTM 中有一行形状转换：

```python
x = x.view(-1, 1, 64)
```

这是为了匹配 PyTorch LSTM 默认输入形状：

```text
[sequence_length, batch_size, feature_dim]
```

在该代码中：

```text
-1 = 序列长度
1  = batch size
64 = 每个状态经过 fc1 后的特征维度
```

普通 PPO 只需要从状态映射到动作概率；PPO-LSTM 则需要先把状态特征整理成序列格式，再输入 LSTM。

## 7. softmax 维度差异

普通 PPO 训练时：

```python
pi = self.pi(s, softmax_dim=1)
```

普通 PPO 的策略输出形状通常是：

```text
[T_horizon, action_dim]
```

例如：

```text
[20, 2]
```

动作维度是第 1 维，因此使用 `dim=1`。

PPO-LSTM 中：

```python
prob = F.softmax(x, dim=2)
```

LSTM 输出形状通常是：

```text
[sequence_length, batch_size, action_dim]
```

例如：

```text
[20, 1, 2]
```

动作维度是第 2 维，因此使用 `dim=2`。这里的原则是：softmax 应该始终作用在动作维度上。

## 8. `v()` 价值函数

### 普通 PPO

```python
def v(self, x):
    x = F.relu(self.fc1(x))
    v = self.fc_v(x)
    return v
```

普通 PPO 的 value 网络只依赖当前状态：

$$
V(s)
$$

### PPO-LSTM

```python
def v(self, x, hidden):
    x = F.relu(self.fc1(x))
    x = x.view(-1, 1, 64)
    x, lstm_hidden = self.lstm(x, hidden)
    v = self.fc_v(x)
    return v
```

LSTM 版的 value 网络依赖当前状态和历史记忆：

$$
V(s, hidden)
$$

也就是说，价值估计不仅参考当前观测，还参考 LSTM 记忆中保留的历史信息。

## 9. `put_data()` 中缓存的数据不同

### 普通 PPO

```python
model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
```

普通 PPO 缓存 6 项：

```text
s       = 当前状态
a       = 动作
r       = 奖励
s_prime = 下一个状态
prob_a  = 旧策略下动作 a 的概率
done    = 是否结束
```

### PPO-LSTM

```python
model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), h_in, h_out, done))
```

LSTM 版额外缓存：

```text
h_in  = 执行动作前的 LSTM 记忆
h_out = 执行动作后的 LSTM 记忆
```

这是 LSTM 版与普通 PPO 的关键差异之一。训练时不仅要知道状态和动作，还要知道当时 LSTM 的记忆状态，否则无法复现该序列在采样时对应的上下文。

## 10. `make_batch()` 中 hidden state 的处理

### 普通 PPO

```python
s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
```

最终返回：

```python
return s, a, r, s_prime, done_mask, prob_a
```

### PPO-LSTM

```python
s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
```

最终返回：

```python
return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
```

这里返回的是：

```text
h_in_lst[0]  = 当前序列开始时的 hidden
h_out_lst[0] = 当前序列第一步后的 hidden
```

因为 LSTM 会沿着序列自动传递 hidden，所以训练时只需要给定序列起点的记忆状态，然后让 LSTM 按时间顺序处理整段状态序列。

更详细的 hidden 对齐解释可参考：[LSTM 训练细节附件](LSTM训练细节附件.md)。

## 11. `train_net()`：PPO 主公式基本不变

普通 PPO 取出 batch：

```python
s, a, r, s_prime, done_mask, prob_a = self.make_batch()
```

然后计算：

```python
td_target = r + gamma * self.v(s_prime) * done_mask
delta = td_target - self.v(s)
```

PPO-LSTM 取出 batch：

```python
s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
```

并构造两组 hidden：

```python
first_hidden  = (h1_in.detach(), h2_in.detach())
second_hidden = (h1_out.detach(), h2_out.detach())
```

其中 `.detach()` 用于切断更早历史的计算图，只保留 hidden 的数值。这样可以避免梯度无限回传到很久以前的采样过程。

## 12. TD target 的差异

### 普通 PPO

```python
td_target = r + gamma * self.v(s_prime) * done_mask
delta = td_target - self.v(s)
```

含义为：

```text
目标价值 = 当前奖励 + gamma * 下一状态价值
TD error = 目标价值 - 当前状态价值
```

### PPO-LSTM

```python
v_prime = self.v(s_prime, second_hidden).squeeze(1)
td_target = r + gamma * v_prime * done_mask
v_s = self.v(s, first_hidden).squeeze(1)
delta = td_target - v_s
```

公式含义不变，但 value 函数多了 hidden：

```text
下一状态价值 = V(s_prime, second_hidden)
当前状态价值 = V(s, first_hidden)
```

也就是说，普通 PPO 估计的是 $V(s)$，而 PPO-LSTM 估计的是带历史记忆的 $V(s, hidden)$。

## 13. Advantage 计算基本一致

普通 PPO：

```python
for delta_t in delta[::-1]:
    advantage = gamma * lmbda * advantage + delta_t[0]
```

PPO-LSTM：

```python
for item in delta[::-1]:
    advantage = gamma * lmbda * advantage + item[0]
```

两段代码本质相同，都是使用 GAE 思路从后往前递推 advantage。LSTM 没有改变 advantage 的数学公式。

## 14. PPO ratio 的差异主要来自张量形状

普通 PPO：

```python
pi = self.pi(s, softmax_dim=1)
pi_a = pi.gather(1, a)
ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
```

PPO-LSTM：

```python
pi, _ = self.pi(s, first_hidden)
pi_a = pi.squeeze(1).gather(1, a)
ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
```

核心公式都是：

$$
ratio =
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

LSTM 版多出的：

```python
pi.squeeze(1)
```

只是为了把形状从：

```text
[sequence_length, 1, action_dim]
```

整理为：

```text
[sequence_length, action_dim]
```

这样才能继续使用 `gather(1, a)` 取出实际动作对应的概率。

## 15. PPO loss 基本一致

普通 PPO：

```python
loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())
```

PPO-LSTM：

```python
loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())
```

两者本质都是：

```text
loss = PPO clipped policy loss + value loss
```

区别在于 LSTM 版提前计算了带 hidden 的状态价值：

```python
v_s = self.v(s, first_hidden).squeeze(1)
```

## 16. 反向传播中的 `retain_graph=True`

普通 PPO：

```python
loss.mean().backward()
```

PPO-LSTM：

```python
loss.mean().backward(retain_graph=True)
```

`retain_graph=True` 表示反向传播后暂时保留计算图。LSTM 版在同一批序列数据上进行多轮更新时，如果计算图被提前释放，可能会触发类似 “Trying to backward through the graph a second time” 的错误。

不过，`retain_graph=True` 会增加显存占用。更稳健的实现通常会更细致地管理 hidden 的 detach、重算前向图以及每轮更新的数据依赖。这里可以先将其理解为 LSTM 训练中为避免计算图释放错误而加入的处理。

## 17. episode 开始时初始化 hidden

普通 PPO：

```python
for n_epi in range(1000):
    s, _ = env.reset()
    done = False
```

普通 PPO 每个 episode 只需要重置环境状态。

PPO-LSTM：

```python
for n_epi in range(10000):
    h_out = (
        torch.zeros([1, 1, 32], dtype=torch.float),
        torch.zeros([1, 1, 32], dtype=torch.float)
    )
    s, _ = env.reset()
    done = False
```

LSTM 需要额外初始化 hidden state。LSTM 的状态由两部分组成：

```text
h = hidden state
c = cell state
```

因此初始化值是一个 tuple：

```python
(h, c)
```

形状为：

```text
[1, 1, 32]
```

分别表示：

```text
1  = LSTM 层数
1  = batch size
32 = hidden size
```

这与网络定义中的 `nn.LSTM(64, 32)` 对应。

## 18. 采样动作时更新 hidden

普通 PPO：

```python
prob = model.pi(torch.from_numpy(s).float())
m = Categorical(prob)
a = m.sample().item()
```

流程为：

```text
状态 s -> 动作概率 prob -> 采样动作 a
```

PPO-LSTM：

```python
h_in = h_out
prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
prob = prob.view(-1)
m = Categorical(prob)
a = m.sample().item()
```

新增逻辑为：

```python
h_in = h_out
prob, h_out = model.pi(..., h_in)
```

这表示上一时刻的记忆 `h_out` 会成为当前时刻的输入记忆 `h_in`，当前状态经过 LSTM 后再产生新的 `h_out`。这就是 LSTM 在交互过程中维护时间连续性的方式。

## 19. `prob.view(-1)` 的作用

LSTM 版在动作采样前执行：

```python
prob = prob.view(-1)
```

因为 `prob` 原本的形状可能是：

```text
[1, 1, 2]
```

而 `Categorical(prob)` 更适合接收：

```text
[2]
```

因此 `view(-1)` 只是形状整理，不代表 PPO 算法发生变化。

## 20. transition 中保存 hidden

普通 PPO：

```python
model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
```

PPO-LSTM：

```python
model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), h_in, h_out, done))
```

这是 PPO-LSTM 的关键改动之一。由于训练时需要重新计算当前策略和价值函数，如果没有保存采样时对应的 hidden，就无法还原当时的序列上下文。

## 21. 训练轮数差异

普通 PPO：

```python
for n_epi in range(1000):
```

PPO-LSTM：

```python
for n_epi in range(10000):
```

LSTM 版训练 episode 更多，可能是因为 LSTM 参数更多、训练更慢，也可能是原始实现希望给模型更多机会学习序列记忆。

如果目标是公平比较普通 PPO 与 PPO-LSTM 的稳定性，可以先统一 episode 数，例如都设为：

```python
for n_epi in range(1000):
```

这样更容易比较两种结构本身带来的差异。

## 22. 可视化差异

普通 PPO 最后会调用：

```python
debugger.plot()
```

PPO-LSTM 版本没有对应可视化逻辑。因此，如果需要比较两者训练稳定性，建议为 LSTM 版也加入：

```python
debugger = DebugTracker(...)
debugger.record_history(...)
debugger.plot()
```

否则只能通过控制台平均分判断训练效果，观察粒度较粗。

## 23. 实现差异汇总

| 代码位置 | 普通 PPO | PPO-LSTM | 本质区别 |
| --- | --- | --- | --- |
| 网络结构 | `fc1 -> pi/v` | `fc1 -> LSTM -> pi/v` | LSTM 版多了记忆层 |
| `fc1` 输出 | 256 | 64 | LSTM 版先降维再输入 LSTM |
| 中间特征 | 256 | LSTM 输出 32 | 策略和值函数都基于 LSTM hidden |
| `pi()` 输入 | `x` | `x, hidden` | LSTM 需要记忆输入 |
| `pi()` 输出 | `prob` | `prob, lstm_hidden` | LSTM 会输出新记忆 |
| `v()` 输入 | `x` | `x, hidden` | value 也参考历史 |
| 数据缓存 | 存 6 项 | 存 8 项 | 多存 `h_in, h_out` |
| batch 处理 | 状态 batch | 序列 batch + hidden | LSTM 要保持时间顺序 |
| TD target | `V(s_prime)` | `V(s_prime, hidden)` | 估值函数带历史记忆 |
| policy ratio | 直接计算 | 先 `squeeze(1)` 再计算 | 主要是张量形状差异 |
| backward | `backward()` | `backward(retain_graph=True)` | LSTM 计算图更复杂 |
| 交互采样 | 每步只用状态 | 每步状态 + hidden | LSTM 逐步更新记忆 |
| 可视化 | 有 `DebugTracker` | 无 | LSTM 版不自动画曲线 |

## 总结

PPO-LSTM 并没有改变 PPO 的核心训练流程。以下部分仍然保持一致：

```text
采样动作
缓存 transition
计算 TD target
计算 advantage
计算 policy ratio
使用 clipped loss
反向传播更新参数
```

真正的变化集中在 hidden state：

```python
prob = model.pi(s)
```

变为：

```python
prob, h_out = model.pi(s, h_in)
```

这意味着 PPO-LSTM 不只是多了一层网络，还必须同步修改数据缓存、batch 构造、价值函数计算和策略前向传播。

可以将两者的区别概括为：

```text
普通 PPO：
根据当前状态直接决定动作。

PPO-LSTM：
根据当前状态和历史记忆共同决定动作。
```

因此，PPO-LSTM 更适合以下场景：

- 当前状态不完整。
- 环境具有部分可观测性。
- 决策需要依赖历史轨迹。
- 状态变化存在明显时间模式。

对于 CartPole 这类观测信息较完整的任务，普通 PPO 已经能够获得较好表现。PPO-LSTM 在这里的主要价值，是展示如何把时间序列记忆机制接入 PPO 的采样、缓存和训练流程。
