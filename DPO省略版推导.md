### 寻找理论上的“完美模型”


幻灯片首先给出了 RLHF 的原始优化目标：

$$\arg \max_\pi \mathbb{E}_{x \sim \mathcal{D}_x, y \sim (\pi(\cdot|x))} r_\phi(x, y) - \beta \text{KL}(\pi \parallel \pi_{\text{ref}})$$

（即：在不偏离参考模型太远的前提下，想尽办法拿到最高的分数。）

数学家们发现，如果我们不考虑神经网络结构的限制（假设策略 $\pi$ 可以是任意模型），这个优化目标其实是有一个闭式解（Closed-form solution）的，也就是理论上最完美的模型 $\pi^*$：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r_\phi(x, y)\right)$$

- **直白解释：** 完美模型生成某句话的概率，等于“参考模型生成的概率”乘以“这句话的得分指数”。得分越高，概率放大的倍数就越大。

- **讨厌的 $Z(x)$：** 这里的 $Z(x)$ 叫配分函数（Partition function）。它的作用是把算出来的值重新压缩回 $0$ 到 $1$ 之间的合法概率分布。**但它是一个极其可怕的项**，因为它要求算出大模型所有可能生成句子的得分总和。在实际工程中，这个计算量是无限大、根本算不出来的。


### 见证奇迹的时刻

既然正着算 $Z(x)$ 算不出来，DPO 的作者们选择了“反向操作”。

**第一步：移项并取对数 (Rearrange & take log)**

作者对上面的完美公式两边同时取对数，把原本用来求 $\pi^*$ 的公式，硬生生反向改写成了用来求奖励模型 $r(x, y)$ 的公式：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

_这也就是前一张幻灯片里“你的语言模型暗地里就是奖励模型”的数学根基。_

**第二步：狸猫换太子 (Parameterization)**

作者把理论上的完美模型 $\pi^*$，直接替换成了我们实际要用神经网络训练的策略模型 $\pi_\theta$。

**第三步：代入 Bradley-Terry 模型 (Substitute into Bradley-Terry)**

这是全篇最天才的一步！还记得我们之前聊过，人类偏好是通过 Bradley-Terry 模型里的**分数差值** $r(x, y_w) - r(x, y_l)$ 来计算的吗？

作者把第一步推导出的 $r(x, y)$ 公式，分别代入到赢家 $y_w$ 和输家 $y_l$ 的得分中，然后相减：

$$( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) ) - ( \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \beta \log Z(x) )$$

**奇迹发生了（The Z will cancel）：**

因为不管模型回答了什么（不管是赢家还是输家），提示词 $x$ 都是同一个。所以那个极其可怕、根本算不出来的常数项 $\beta \log Z(x)$ **被完美地减掉、抵消了！**
