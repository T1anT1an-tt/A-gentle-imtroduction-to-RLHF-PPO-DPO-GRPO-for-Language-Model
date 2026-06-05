# DPO 省略版推导

本文整理 DPO（Direct Preference Optimization）中最关键的一段推导：如何从带 KL 约束的 RLHF 优化目标，得到不需要显式训练奖励模型的偏好学习目标。

## 1. 从 RLHF 的 KL 约束目标出发

RLHF 中常见的策略优化目标可以写为：

$$
\arg \max_\pi
\mathbb{E}_{x \sim \mathcal{D}_x,\, y \sim \pi(\cdot \mid x)}
\left[
r_\phi(x, y)
- \beta \operatorname{KL}
\left(
\pi(\cdot \mid x)
\parallel
\pi_{\text{ref}}(\cdot \mid x)
\right)
\right]
$$

其中：

- $x$ 表示 prompt。
- $y$ 表示模型回答。
- $r_\phi(x,y)$ 表示奖励模型给回答打出的分数。
- $\pi_{\text{ref}}$ 表示参考模型，通常是 SFT 模型。
- $\beta$ 控制策略偏离参考模型的惩罚强度。

这个目标的含义是：模型既要生成高奖励回答，又不能偏离参考模型太远。

## 2. 理论最优策略的闭式解

如果暂时不考虑神经网络结构的限制，并允许策略 $\pi$ 是任意概率分布，那么上述目标存在一个理论闭式解：

$$
\pi^*(y \mid x)
=
\frac{1}{Z(x)}
\pi_{\text{ref}}(y \mid x)
\exp \left(
\frac{1}{\beta} r_\phi(x,y)
\right)
$$

这个式子可以理解为：

```text
理论最优策略的概率
= 参考模型概率 × 奖励分数带来的指数加权
```

如果某个回答的奖励更高，$\exp(\frac{1}{\beta}r_\phi(x,y))$ 会放大它的生成概率；如果奖励更低，它的概率就会相对变小。

式子中的 $Z(x)$ 是配分函数（partition function）：

$$
Z(x)
=
\sum_y
\pi_{\text{ref}}(y \mid x)
\exp \left(
\frac{1}{\beta} r_\phi(x,y)
\right)
$$

$Z(x)$ 的作用是归一化，使所有候选回答的概率之和等于 1。问题在于，对于大语言模型来说，所有可能回答 $y$ 的空间极大，因此 $Z(x)$ 在实际训练中无法直接计算。

## 3. 将奖励函数反解出来

从闭式解出发，对两边取对数：

$$
\log \pi^*(y \mid x)
=
\log \pi_{\text{ref}}(y \mid x)
+ \frac{1}{\beta}r_\phi(x,y)
- \log Z(x)
$$

整理可得：

$$
r_\phi(x,y)
=
\beta
\log
\frac{\pi^*(y \mid x)}
{\pi_{\text{ref}}(y \mid x)}
+ \beta \log Z(x)
$$

这一步是 DPO 的核心观察：在理论最优条件下，奖励函数可以由策略模型和参考模型的 log-prob ratio 表示。

也就是说，语言模型本身可以隐式表达一个奖励函数：

```text
回答相对于参考模型越被当前策略偏好，
对应的隐式奖励就越高。
```

## 4. 用可训练策略替代理论最优策略

真实训练中无法直接得到 $\pi^*$，因此 DPO 将理论最优策略重新参数化为当前要训练的策略模型 $\pi_\theta$：

$$
r_\theta(x,y)
=
\beta
\log
\frac{\pi_\theta(y \mid x)}
{\pi_{\text{ref}}(y \mid x)}
+ \beta \log Z(x)
$$

这里的 $r_\theta(x,y)$ 不是额外训练出来的奖励模型，而是由策略模型 $\pi_\theta$ 和参考模型 $\pi_{\text{ref}}$ 共同定义的隐式奖励。

## 5. 代入 Bradley-Terry 偏好模型

偏好数据通常由三元组组成：

$$
(x, y_w, y_l)
$$

其中 $y_w$ 是人类更偏好的回答，$y_l$ 是较差回答。Bradley-Terry 模型用两者的奖励差来表示 $y_w$ 胜过 $y_l$ 的概率：

$$
P(y_w \succ y_l \mid x)
=
\sigma
\left(
r(x,y_w) - r(x,y_l)
\right)
$$

将上面的隐式奖励代入奖励差：

$$
r_\theta(x,y_w) - r_\theta(x,y_l)
$$

得到：

$$
\left(
\beta \log
\frac{\pi_\theta(y_w \mid x)}
{\pi_{\text{ref}}(y_w \mid x)}
+ \beta \log Z(x)
\right)
-
\left(
\beta \log
\frac{\pi_\theta(y_l \mid x)}
{\pi_{\text{ref}}(y_l \mid x)}
+ \beta \log Z(x)
\right)
$$

由于 $y_w$ 和 $y_l$ 来自同一个 prompt $x$，所以 $\beta \log Z(x)$ 是同一个常数项，在相减时会完全抵消：

$$
r_\theta(x,y_w) - r_\theta(x,y_l)
=
\beta
\left[
\log
\frac{\pi_\theta(y_w \mid x)}
{\pi_{\text{ref}}(y_w \mid x)}
-
\log
\frac{\pi_\theta(y_l \mid x)}
{\pi_{\text{ref}}(y_l \mid x)}
\right]
$$

这一步解释了 DPO 为什么可以绕过无法计算的 $Z(x)$。

## 6. 得到 DPO 损失

将奖励差代回 Bradley-Terry 模型，并对偏好数据做负对数似然，就得到 DPO 的训练目标：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}})
=
-
\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}
\left[
\log
\sigma
\left(
\beta
\left[
\log
\frac{\pi_\theta(y_w \mid x)}
{\pi_{\text{ref}}(y_w \mid x)}
-
\log
\frac{\pi_\theta(y_l \mid x)}
{\pi_{\text{ref}}(y_l \mid x)}
\right]
\right)
\right]
$$

从形式上看，DPO 直接优化偏好对：

- 提高 preferred response $y_w$ 相对于参考模型的概率比。
- 降低 rejected response $y_l$ 相对于参考模型的概率比。
- 通过 $\beta$ 控制策略偏离参考模型的幅度。

## 总结

DPO 推导的关键链条是：

```text
RLHF KL 约束目标
-> 理论最优策略的闭式解
-> 反解出奖励函数
-> 用 pi_theta 替换 pi*
-> 代入 Bradley-Terry 偏好模型
-> Z(x) 在偏好差中抵消
-> 得到可直接训练语言模型的 DPO loss
```

因此，DPO 的核心价值在于：它把“先训练奖励模型，再用 RL 优化策略”的流程，改写成了一个直接基于偏好对训练策略模型的监督式目标。
