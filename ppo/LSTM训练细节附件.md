# PPO-LSTM 中 `first_hidden` 与 `second_hidden` 的含义

在 PPO-LSTM 实现中，`first_hidden` 和 `second_hidden` 不是按英文序号区分“第一个 hidden”和“第二个 hidden”，而是分别对应两组不同的状态序列：

```text
first_hidden  = 用于计算当前状态序列 s 的初始 LSTM 记忆
second_hidden = 用于计算下一状态序列 s_prime 的初始 LSTM 记忆
```

因此，核心对应关系是：

```text
first_hidden  -> s
second_hidden -> s_prime
```

这两个 hidden 的作用，是保证在计算 $V(s)$ 和 $V(s')$ 时，状态序列与 LSTM 记忆在时间步上正确对齐。

## 1. 相关代码

典型代码如下：

```python
first_hidden  = (h1_in.detach(), h2_in.detach())
second_hidden = (h1_out.detach(), h2_out.detach())

for i in range(K_epoch):
    v_prime = self.v(s_prime, second_hidden).squeeze(1)
    td_target = r + gamma * v_prime * done_mask
    v_s = self.v(s, first_hidden).squeeze(1)
    delta = td_target - v_s
    delta = delta.detach().numpy()
```

这段代码的目标是计算 TD error：

```text
td_target = r + gamma * V(s_prime)
delta     = td_target - V(s)
```

普通 PPO 中，价值网络只需要输入状态：

```python
td_target = r + gamma * self.v(s_prime) * done_mask
delta = td_target - self.v(s)
```

但在 PPO-LSTM 中，价值网络不仅依赖当前状态，还依赖历史记忆。因此 `v()` 的输入从普通 PPO 的 `x` 变成了 `x, hidden`：

```python
v_prime = self.v(s_prime, second_hidden)
v_s     = self.v(s, first_hidden)
```

## 2. 为什么计算 `s` 时使用 `first_hidden`

假设采样得到一段长度为 20 的轨迹：

```text
s0, s1, s2, s3, ..., s19
```

LSTM 在真实交互中的状态推进方式可以表示为：

```text
s0 + h0 -> h1
s1 + h1 -> h2
s2 + h2 -> h3
s3 + h3 -> h4
...
```

对于当前状态序列 `s = [s0, s1, s2, ..., s19]`，序列起点是 `s0`，因此计算这段序列的价值时，应该使用 `s0` 之前的 hidden，即 `h0`。

代码中的：

```python
first_hidden = (h1_in.detach(), h2_in.detach())
```

就表示这段 `s` 序列开始之前的 LSTM 记忆。随后：

```python
v_s = self.v(s, first_hidden).squeeze(1)
```

表示以 `first_hidden` 为初始记忆，重新运行 `s0, s1, ..., s19`，并得到每个时间步对应的 $V(s)$。

## 3. 为什么计算 `s_prime` 时使用 `second_hidden`

如果：

```text
s = [s0, s1, s2, ..., s19]
```

那么下一状态序列是：

```text
s_prime = [s1, s2, s3, ..., s20]
```

也就是说，`s_prime` 的第一个元素不是 `s0`，而是 `s1`。

对应到 LSTM hidden：

```text
s0 之前的 hidden 是 h0
s1 之前的 hidden 是 h1
s2 之前的 hidden 是 h2
s3 之前的 hidden 是 h3
```

因此，如果要计算：

```text
V(s1), V(s2), V(s3), ...
```

就不能从 `h0` 开始，而应该从 `h1` 开始。这里的 `h1` 正是模型处理完 `s0` 之后得到的 hidden，也就是代码中的 `h_out`。

所以：

```python
second_hidden = (h1_out.detach(), h2_out.detach())
```

表示 `s_prime` 序列开始之前的 LSTM 记忆。随后：

```python
v_prime = self.v(s_prime, second_hidden).squeeze(1)
```

表示以 `second_hidden` 为初始记忆，重新运行 `s1, s2, ..., s20`，并得到每个时间步对应的 $V(s')$。

## 4. 时间步对齐关系

可以用下表理解 `s`、`s_prime` 和 hidden 的对齐方式：

```text
时间步:        t0        t1        t2        t3

状态:          s0        s1        s2        s3
hidden 前:     h0        h1        h2        h3
hidden 后:     h1        h2        h3        h4

s 序列:        s0        s1        s2        s3
起点 hidden:   h0
使用 hidden:   first_hidden

s_prime 序列: s1        s2        s3        s4
起点 hidden:   h1
使用 hidden:   second_hidden
```

这就是为什么 `V(s)` 和 `V(s_prime)` 不能使用同一个初始 hidden。它们对应的状态序列起点不同，因此所需的历史记忆也不同。

## 5. 为什么不能都使用 `first_hidden`

如果把代码写成：

```python
v_prime = self.v(s_prime, first_hidden)
```

那么实际含义会变成：

```text
s_prime = [s1, s2, s3, ...]
起点 hidden = h0
```

这与真实交互过程不一致。模型在真实看到 `s1` 时，已经处理过 `s0`，此时 hidden 应该是 `h1`，而不是 `h0`。

如果 `s_prime` 也使用 `first_hidden`，就等价于让 LSTM 在计算 `s1` 时忽略 `s0` 已经带来的历史信息，从而破坏时间顺序。对于依赖历史信息的 LSTM 价值网络，这会导致 $V(s')$ 的估计与实际轨迹不对齐。

## 6. `.detach()` 的作用

代码中对 hidden 使用了 `.detach()`：

```python
first_hidden  = (h1_in.detach(), h2_in.detach())
second_hidden = (h1_out.detach(), h2_out.detach())
```

`.detach()` 的作用是保留 hidden 的数值，但切断它之前的计算图。

LSTM 的 hidden 是网络前向传播得到的张量。如果不使用 `.detach()`，PyTorch 会继续追踪 hidden 的历史来源，并把计算图一路连接到更早的时间步。随着采样序列不断累积，计算图会越来越长，带来额外显存开销，也会使梯度传播超出当前训练片段。

在 PPO-LSTM 中，通常只希望对当前 rollout 片段进行反向传播，而不是让梯度继续回传到更早的采样过程。因此 `.detach()` 可以理解为：

```text
保留当前 hidden 的数值，
但不让梯度继续穿过 hidden 回到更早的历史。
```

这是一种截断反向传播（truncated backpropagation through time）的处理方式。

## 7. `h1_in, h2_in` 中的 `h1` 和 `h2`

变量名 `h1_in`、`h2_in`、`h1_out`、`h2_out` 容易让人误解为“第一步 hidden”和“第二步 hidden”。在 LSTM 中，它们更准确的含义是两个内部状态：

```text
h1 = hidden state
h2 = cell state
```

更清晰的命名方式通常是：

```python
(h_in, c_in)
(h_out, c_out)
```

如果重命名，则这段代码可以写得更直观：

```python
hidden_for_s = (h_in.detach(), c_in.detach())
hidden_for_s_prime = (h_out.detach(), c_out.detach())
```

对应关系为：

```text
first_hidden  = hidden_for_s
second_hidden = hidden_for_s_prime
```

## 8. 逐行解释

```python
first_hidden = (h1_in.detach(), h2_in.detach())
```

取出当前状态序列 `s` 开始前的 LSTM 记忆，作为计算 $V(s)$ 的初始 hidden。

```python
second_hidden = (h1_out.detach(), h2_out.detach())
```

取出当前状态序列第一步之后的 LSTM 记忆，作为计算 $V(s')$ 的初始 hidden。

```python
v_prime = self.v(s_prime, second_hidden).squeeze(1)
```

使用与 `s_prime` 对齐的 hidden，计算下一状态价值 $V(s')$。

```python
td_target = r + gamma * v_prime * done_mask
```

使用即时奖励 $r$ 和下一状态价值 $V(s')$ 构造 TD target。

```python
v_s = self.v(s, first_hidden).squeeze(1)
```

使用与 `s` 对齐的 hidden，计算当前状态价值 $V(s)$。

```python
delta = td_target - v_s
```

计算 TD error。它也是后续构造 advantage 的基础。

## 总结

可以将这段逻辑概括为：

```text
first_hidden:
用于 s，因为 s 从 s0 开始，需要 s0 之前的 h0。

second_hidden:
用于 s_prime，因为 s_prime 从 s1 开始，需要 s1 之前的 h1。

detach:
切断旧计算图，只保留 hidden 的当前数值。
```

这段代码没有改变 PPO 的 TD target 公式。它的主要作用是让 LSTM 版 PPO 在计算 $V(s)$ 和 $V(s')$ 时，保持状态序列与 hidden 记忆的时间位置一致。
