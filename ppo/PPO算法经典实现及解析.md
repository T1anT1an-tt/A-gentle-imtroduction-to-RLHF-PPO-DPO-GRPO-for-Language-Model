
![[Pasted image 20260604135228.png]]

```Python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 超参数设置 (Hyperparameters)
# ==========================================
learning_rate = 0.0005  # 优化器的学习率，决定每次更新参数的步子大小
gamma         = 0.98    # 折扣因子 (Discount factor)，决定对未来奖励的重视程度 (0~1)
lmbda         = 0.95    # GAE (广义优势估计) 的平滑参数，平衡方差和偏差
eps_clip      = 0.1     # PPO 的核心截断范围 (通常设为 0.1 或 0.2)，防止更新幅度过大
K_epoch       = 3       # 每次收集完一批数据后，用来重复训练神经网络的次数 (复用数据提高效率)
T_horizon     = 20      # 视野边界 (Rollout Length)，连走多少步之后停下来算一次总账 (等同于极简版 Batch Size)

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []  # 用来充当“小本本”，存储每次与环境交互收集到的经验数据

        # 共享的底层特征提取网络
        self.fc1   = nn.Linear(4,256)
        # 演员 (Actor) / 策略模型：输出采取各个动作的概率 (CartPole 只有 2 个动作：向左/向右)
        self.fc_pi = nn.Linear(256,2)
        # 军师 (Critic) / 价值模型：输出当前状态的预期总得分 (一个具体的分数值)
        self.fc_v  = nn.Linear(256,1)
        # 优化器，负责最终执行反向传播，真正去修改网络里的参数
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # 演员的判断：输出动作的概率分布
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        # 用 softmax 把神经网络输出的数值转换成加起来等于 1 的概率分布
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    # 军师的预判：评估当前局面的价值
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    # 收集经验：把走一步的所见所为存进小本本
    def put_data(self, transition):
        self.data.append(transition)

    # 打包数据：把小本本里的单条记录，打包成 PyTorch 可以批量处理的张量 (Tensor)
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)               # 当时所在的状态
            a_lst.append([a])             # 当时采取的动作
            r_lst.append([r])             # 当时得到的真实奖励
            s_prime_lst.append(s_prime)   # 动作执行后的下一个状态
            prob_a_lst.append([prob_a])   # 【重点】记录当时采取这个动作的概率 (作为旧策略 P_old)
            done_mask = 0 if done else 1  # 游戏是否结束的掩码 (结束了就是 0，切断对未来的预估)
            done_lst.append([done_mask])

        # 转换成模型计算需要的 Tensor 格式
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = [] # 经验打包完后立刻清空小本本，准备下一轮的收集 (On-policy 的体现)
        return s, a, r, s_prime, done_mask, prob_a

    # ==========================================
    # PPO 最核心的算法逻辑：复盘与学习
    # ==========================================
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # 把这批收集到的经验反复复用，训练 K_epoch 次
        for i in range(K_epoch):
            # 1. 计算 TD 目标值 = 当前拿到的真实奖励 + 军师对下个状态的预估 (自举 Bootstrapping)
            td_target = r + gamma * self.v(s_prime) * done_mask
            # 2. 计算 TD 误差 = 现实(TD 目标值) - 预期(军师一开始对当前状态的预估)
            delta = td_target - self.v(s)
            delta = delta.detach().numpy() # 剥离出计算图，把它当做固定的常数来算，不参与梯度反向传播

            # 3. 计算 GAE (广义优势估计)：给动作一个兼顾短期与长期的综合评分
            advantage_lst = []
            advantage = 0.0
            # 巧妙的动态规划：从后往前逆向推算优势
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse() # 算完之后反转回来，恢复正常的时间顺序
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # 4. 计算新旧策略概率比 (Ratio)
            pi = self.pi(s, softmax_dim=1)  # 获取模型现在的脑子在相同状态下给出的动作概率分布
            pi_a = pi.gather(1,a)           # 提取出当时采取的那个特定动作现在的概率
            # 利用数学公式 ln(a/b) = ln(a) - ln(b) 稳定计算除法：(新概率 / 旧概率)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            # 5. PPO 核心魔法：计算截断目标函数 (Clipped Surrogate Objective)
            surr1 = ratio * advantage
            # 把 ratio 死死限制在 [1-eps_clip, 1+eps_clip] (比如 0.9 到 1.1) 之间
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage

            # 6. 计算最终的总 Loss (损失值)
            # 左半边：取 surr1 和 surr2 较小的值加负号 (PyTorch默认求最小，加负号等于求最大优势，用来训练 Actor)
            # 右半边：用平滑 L1 损失，让军师的预估值逼近真实结果 td_target (用来训练 Critic)
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            # 7. 经典的 PyTorch 参数更新三步曲
            self.optimizer.zero_grad() # 清除上一次的梯度记忆
            loss.mean().backward()     # 根据 Loss 沿着神经网络向后求导，计算当前梯度
            self.optimizer.step()      # 优化器发力，真正修改底层和 Actor/Critic 里的参数权重

def main():
    env = gym.make('CartPole-v1') # 创建测试环境 (推车杆游戏)
    model = PPO()                 # 实例化我们刚定义的大脑
    score = 0.0
    print_interval = 20           # 每训练 20 局打印一次成绩

    # 让智能体玩 10000 局游戏
    for n_epi in range(10000):
        s, _ = env.reset() # 游戏开始，环境复位，获取初始状态 s
        done = False
        while not done:
            # 连续走 T_horizon 步 (这 20 步期间只存数据，绝对不更新参数！)
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float()) # 1. 演员看状态，出概率分布
                m = Categorical(prob)                        # 2. 把概率分布包装成可采样的对象
                a = m.sample().item()                        # 3. 根据概率“掷骰子”决定具体动作 (这步就是 Sampling!)

                # 4. 智能体将动作发给环境执行，环境反馈：新状态、真实奖励、游戏是否结束
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                # 5. 把这一次“互动”产生的所有信息打包装进小本本 (奖励缩小 100 倍是为了稳定训练数值)
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime # 视角推移，当前状态更新为下一个状态

                score += r
                if done:
                    break # 如果游戏死了或者通关了，提前跳出 T_horizon 的收集循环

            # 当小本本收集满了 T_horizon 步的经验，暂停游戏，开始复盘并更新大脑参数！
            model.train_net()

        # 定期打印成绩汇报进度
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
```

### PPO 算法核心公式

PPO 的目标是让模型学会在特定状态下采取最优动作。在 `train_net` 更新参数时，它在背后计算着这四个核心公式：

**1. 现实与预期的落差：TD 误差 (Temporal Difference Error)**

模型在走完一步后，需要评估这一步到底好不好。我们需要计算“实际得到的回报”与“原来预估的回报”之间的差值。

$$V_{target} = r_t + \gamma V(s_{t+1})$$

$$\delta_t = V_{target} - V(s_t)$$

- $V_{target}$ 是目标价值：等于当前这一步拿到的真实奖励 $r_t$，加上对未来状态的价值预测 $\gamma V(s_{t+1})$。

- $\delta_t$ 是 TD 误差：如果它大于 0，说明这一步的实际结果比模型预期的要好，是个惊喜。


**2. 统筹全局的评分：广义优势估计 (GAE, Generalized Advantage Estimation)**

单看一步的 $\delta_t$ 太短视了，PPO 会使用 GAE 将未来的误差也折算进来，给这个动作一个综合评分（优势 $A_t$）。

$$A_t = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots$$

- 其中 $\gamma$ 和 $\lambda$ 都是衰减系数（0 到 1 之间）。越往后的未来误差，对当前动作的影响越小。


**3. 新旧策略的差异：概率比值 (Probability Ratio)**

PPO 在更新策略时，需要对比“更新后的新脑子”和“更新前的旧脑子”在同样状态下做出同样动作的概率变化。（与经验回放做区分 一个是不同策略下采样的适配问题 一个是新旧策略下的采样（当前））

$$ratio_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

- 如果 $ratio_t > 1$，说明新策略比旧策略更倾向于采取这个动作。


**4. 核心魔法：截断的替代目标函数 (Clipped Surrogate Objective)**

这是 PPO 稳定性的基石。我们希望最大化优势（好动作多做，坏动作少做），但又不能让模型步子迈得太大扯到蛋。（gemini中文语料库到底有什么毛病）

$$L^{CLIP} = \min(ratio_t \cdot A_t, \text{clip}(ratio_t, 1-\epsilon, 1+\epsilon) \cdot A_t)$$

- 公式用 $\epsilon$（通常是 0.1 或 0.2）把 $ratio_t$ 截断在 $[0.8, 1.2]$ 的范围内。这意味着，就算某个动作特别好，单次更新时概率也不要翻倍地涨。

----
# 调参日志
## 运行原始版本
```
of episode :20, avg score : 20.8
of episode :40, avg score : 26.1
of episode :60, avg score : 39.3
of episode :80, avg score : 33.3
of episode :100, avg score : 82.8
of episode :120, avg score : 151.6
of episode :140, avg score : 206.8
of episode :160, avg score : 201.2
of episode :180, avg score : 349.9
of episode :200, avg score : 219.8
of episode :220, avg score : 225.5
of episode :240, avg score : 338.6
of episode :260, avg score : 432.6
of episode :280, avg score : 404.8
of episode :300, avg score : 434.9
of episode :320, avg score : 409.2
of episode :340, avg score : 472.9
of episode :360, avg score : 484.1
of episode :380, avg score : 413.9
of episode :400, avg score : 394.3
of episode :420, avg score : 324.7
of episode :440, avg score : 222.3
of episode :460, avg score : 355.1
of episode :480, avg score : 364.3
of episode :500, avg score : 381.3
of episode :520, avg score : 401.9
of episode :540, avg score : 396.1
of episode :560, avg score : 378.2
of episode :580, avg score : 464.3
of episode :600, avg score : 355.6
of episode :620, avg score : 373.6
of episode :640, avg score : 192.7
of episode :660, avg score : 226.8
of episode :680, avg score : 281.4
of episode :700, avg score : 411.8
of episode :720, avg score : 362.6
of episode :740, avg score : 352.4
of episode :760, avg score : 338.9
of episode :780, avg score : 13.8
of episode :800, avg score : 9.2
of episode :820, avg score : 9.6
of episode :840, avg score : 9.4
of episode :860, avg score : 9.6
of episode :880, avg score : 9.5
of episode :900, avg score : 9.3
of episode :920, avg score : 9.4
of episode :940, avg score : 9.5
of episode :960, avg score : 9.3
of episode :980, avg score : 9.3
```
### 为什么分数在第 780 局突然崩塌？
终端日志里有一个非常典型的强化学习现象：
- 前期学习很好 ：分数从 20 多分一路涨到了 480 多分，这说明 PPO 的模型架构和环境交互是没有问题的，它确实学会了怎么让倒立摆保持平衡。
- 后期突然崩塌 ：在 # of episode :780 的时候，平均分突然从 338 暴跌到了 13.8，并且之后一直维持在 9 分左右（完全变成了随机乱动，甚至比随机还差）。
这种现象在 PPO 里叫 “灾难性遗忘” (Catastrophic Forgetting) 或者 “策略崩塌” (Policy Collapse) 。造成这个现象的常见原因有几个：
- 学习率（Learning Rate）偏大 ： ppo.py 中目前的学习率是 0.0005 。在模型已经学得差不多（快满分）的时候，太大的学习率会导致模型在某一次更新时步子迈得太大，直接把网络参数更新到了一个极端的“坏”区域。
- 没有学习率衰减（Learning Rate Annealing） ：PPO 通常建议在训练后期逐渐降低学习率，但这份精简版代码里是固定学习率。
- Clip 参数（ eps_clip = 0.1 ）太小或太大 ：如果裁剪限制得不够好，单次更新偏离过大，策略就会崩溃。
怎么解决？ 最简单的办法是 调小学习率 （比如改成 0.0001 ）或者增加学习率衰减。不过既然这是一个极简版的学习代码，偶尔出现崩塌是很正常的，它证明了**算法的敏感性**。(实测代码不变的情况下 确实只是有时候会出现这种概率)
----

## 微调实验
实验设置见[[强化学习/zhaoshiyu/project/minimalRL_Tt/ppo/README|README]]
1. mean_reward_curve.png

- 含义：每个超参数设置下， ma_50 ，也就是回报的 50 回合滑动平均曲线。
- 好的表现：曲线更高、上升更快、后期更平稳。
- 不好的表现：长期很低、振荡很大、后期掉下来。
- 比较方法：谁在后半段长期更高，谁通常更优；如果两条线均值差不多，就看阴影谁更窄。
![[Pasted image 20260604154422.png|693]]
![[Pasted image 20260604154458.png|692]]
![[Pasted image 20260604155044.png|692]]
![[Pasted image 20260604155135.png|692]]
2. final_100_return_boxplot.png

- 含义：取最后 100 个 episode 的 return ，看不同超参数下最终表现分布。
- 盒子中位数越高越好。
- 盒子越矮、须越短，说明波动越小、稳定性越好。
- 离群点很多，说明偶尔会特别好或特别差，稳定性一般。
- 如果某个参数中位数高，但箱体特别高，说明“平均强，但不稳”。
![[Pasted image 20260604155415.png]]
![[Pasted image 20260604155440.png|695]]
![[Pasted image 20260604155504.png]]
![[Pasted image 20260604155528.png]]
3. policy_entropy.png

- 含义：策略熵，衡量策略随机性/探索程度。
- 怎么看：
- 前期高一些通常正常，表示还在探索。
- 随训练推进逐渐下降，通常说明策略在收敛。
- 如果很快掉到很低，可能探索不足，容易早熟收敛到次优策略。
- 如果一直很高，可能策略始终不确定，学不稳。
- 结合 reward 看：高 entropy 但 reward 不涨，说明“在乱试”；entropy 逐步下降且 reward 上升，通常是健康训练。
![[Pasted image 20260604155639.png]]
![[Pasted image 20260604155708.png]]
![[Pasted image 20260604155721.png]]
![[Pasted image 20260604155740.png]]


4. actor_loss.png

- 含义：策略网络的损失。
- 怎么看：
- 这张图一般不直接看“越低越好”，因为 PPO 的 loss 定义不是简单监督学习损失。
- 主要看是否平稳、是否有异常尖峰。
- 如果震荡特别剧烈，常意味着学习率太大、更新太猛或训练不稳定。
- 如果长期几乎不动，可能更新太弱，策略学不起来。
- 结论一般不能只靠这张图下，要结合 reward、ratio、clip_fraction 一起看。
![[Pasted image 20260604155843.png]]
![[Pasted image 20260604155856.png]]
![[Pasted image 20260604155904.png]]
![[Pasted image 20260604155911.png]]

4. critic_loss.png

- 含义：价值网络损失，反映 value function 拟合回报的难度。
- 怎么看：
- 一般希望逐步下降或至少保持在较稳定范围内。
- 如果特别大且持续抖动，说明 value 学不好，优势估计也会受影响。
- 如果 critic_loss 爆炸，通常训练整体也会不稳。
- 但“很低”也不必然代表策略一定好，仍然要回到 reward 判断。
![[Pasted image 20260604155949.png]]
![[Pasted image 20260604160002.png]]
![[Pasted image 20260604160010.png]]
![[Pasted image 20260604160019.png]]
6. total_loss.png

- 含义：总损失，通常是 actor loss、critic loss 以及其他正则项组合。
- 怎么看：
- 主要看整体训练是否出现异常震荡、发散、突然跳变。
- 如果 total_loss 经常剧烈尖峰，往往训练过程不健康。
- 如果它比较平稳，而 reward 也稳步提升，通常是好现象。
- 单独价值有限，适合作为“训练健康度监控图”。
![[Pasted image 20260604160146.png]]
![[Pasted image 20260604160155.png]]
![[Pasted image 20260604160217.png]]
![[Pasted image 20260604160340.png]]

7. ratio_mean.png

- 含义：PPO 中新旧策略概率比值 r_t 的平均值。
- 怎么看：
- 理想情况下，ratio mean 应该在 1 附近波动。
- 明显偏离 1 很多，说明一次更新把策略改动得太大。
- 如果长期高于 1 很多或低于 1 很多，通常表示更新过激，可能不稳定。
- 如果一直特别接近 1 且 reward 也不涨，可能更新太保守。
![[Pasted image 20260604160413.png]]
![[Pasted image 20260604160425.png]]
![[Pasted image 20260604160443.png]]
![[Pasted image 20260604160455.png]]



8. ratio_std.png

- 含义：概率比值的标准差，反映不同样本上策略更新幅度分散程度。
- 怎么看：
- 太大：说明有些样本更新很猛，有些很小，训练可能不稳。
- 适中：通常更健康。
- 太小：可能更新过于保守，策略变化不足。
- 一般来说，reward 高且 ratio_std 不夸张，是比较理想的。
![[Pasted image 20260604160527.png]]
![[Pasted image 20260604160534.png]]
![[Pasted image 20260604160545.png]]
![[Pasted image 20260604160552.png]]


9. clip_fraction.png

- 含义：有多少比例的样本在 PPO 更新时被 clipping 截断了。
- 怎么看：
- 太低，接近 0：可能更新太小，clip 几乎没起作用，学习可能偏慢。
- 适中：通常最好，说明更新有力度但没有过猛。
- 太高：说明大量样本都被裁剪，策略更新过大，可能不稳定。
- 如果某组参数 reward 不好且 clip_fraction 很高，常见原因就是“步子迈太大”。
![[Pasted image 20260604160630.png]]
![[Pasted image 20260604160658.png]]
![[Pasted image 20260604160705.png]]
![[Pasted image 20260604160722.png]]

10. advantage_mean.png

- 含义：优势函数均值，反映样本整体相对基线的好坏趋势。
- 怎么看：
- 理想情况下，优势通常应围绕某个合理范围，不应长期异常漂移。
- 如果数值剧烈震荡，说明估计噪声较大，训练信号可能不稳定。
- 如果长期极端偏正或偏负，也值得警惕，可能是 value 估计或归一化有问题。
- 这张图更多是诊断图，不是最终效果图，重点看“稳不稳”。
![[Pasted image 20260604160756.png]]
![[Pasted image 20260604160815.png|697]]![[Pasted image 20260604160853.png]]
![[Pasted image 20260604160928.png]]


## 微调实验结论
**总体判断**
- 我看完总表 [summary_results.csv](file:///root/autodl-tmp/ppo/results/summary_results.csv) 后的结论是：这批 PPO 实验已经明显学到了，但“稳定求解”只在 `learning_rate=0.001` 上部分出现，整体还没到“稳稳解决任务”的程度。
- 默认主配置基本是 `lr=0.0005`、`eps_clip=0.1`、`T_horizon=20`、三种 trick 全开，可见于代表性配置 [config.json](file:///root/autodl-tmp/ppo/results/lr_sensitivity/learning_rate_0_001/seed_0/config.json)；这套配置能学到中高回报，但 5 个 seed 里 `0/5` 达到求解判据。
- 真正最强的单项因素是学习率：`lr=0.001` 的平均 `final_100_mean=350.08`，而且 `3/5` 个 seed 曾达到 solve 阈值；相比之下 `lr=0.0005` 平均只有 `203.88`，`lr=0.0001` 几乎没学起来，只有 `26.97`。
- 第二个最关键因素是 `T_horizon`：`20` 最好，`64` 明显退化，`128/256` 基本崩掉，说明这个任务里 rollout 太长会严重伤害训练效果。
- `eps_clip` 反而不敏感：`0.05/0.1/0.2/0.3` 的均值都在 `199~207` 左右，说明你当前实现和任务下，clip 范围不是主矛盾。

**关键结论**
- 如果你问“这次实验最重要的结论是什么”，答案是：`lr` 和 `T_horizon` 决定成败，`eps_clip` 只是在可接受区间内微调。
- 如果你问“当前最好配置是什么样子”，答案是：`learning_rate=0.001` 明显最好，但它是“有能力冲到 solve 区域”，不是“所有 seed 都稳定保持 solve”。
- 如果你问“默认 PPO + tricks 全开够不够”，答案是：不够稳。它比 baseline 强，但还没有形成稳健的跨 seed 成功。
- 如果你问“哪些 trick 真有帮助”，答案是：`advantage normalization` 最有价值；`entropy` 单独开效果最差；`grad clip` 这批结果里几乎没体现出收益。

**分项解读**
- 学习率敏感性：
  `0.0001 -> 26.97`，`0.0003 -> 64.26`，`0.0005 -> 203.88`，`0.001 -> 350.08`。这是最清晰的一条曲线，说明你的实现更需要偏大的学习率才能快速进入有效学习区间。
- Horizon 敏感性：
  `T=20 -> 203.88`，`T=64 -> 78.32`，`T=128 -> 40.51`，`T=256 -> 26.95`。随着 horizon 增大，性能几乎单调恶化，这个趋势非常强，结论可信度很高。
- Clip 敏感性：
  `0.05 -> 204.23`，`0.1 -> 203.88`，`0.2 -> 206.86`，`0.3 -> 199.43`。差距非常小，说明你现在不需要把主要精力花在 clip 上。
- Ablation：
  `adv_norm -> 211.44`，`all -> 203.88`，`baseline -> 161.42`，`grad_clip -> 161.42`，`entropy -> 113.81`。这说明 `adv_norm` 是最关键 trick，`entropy` 在当前设置下反而拖后腿。
- 一个很值得注意的点是：`baseline` 和 `grad_clip` 的 5 个 seed 结果完全一样，这很不寻常。它可能表示梯度裁剪在这些运行里几乎从未触发，也可能表示实验分组或记录有重复，建议专门核查。

**稳定性判断**
- 虽然 `lr=0.001` 有 `3/5` 个 seed 记录了 `time_to_solve`，但这不等于训练结束时仍然稳定 solve。
- 求解判据在 [train.py:L81-L128](file:///root/autodl-tmp/ppo/train.py#L81-L128) 里写得很清楚：首次满足最近 `50` 个 episode 的均值 `ma_50 >= 475` 时，记录 `time_to_solve`。
- 我继续检查了 `lr=0.001` 的 5 个 seed：`seed_3` 训练末尾还能保持 `last_ma50=491.54`，这是最扎实的成功；但 `seed_0` 和 `seed_4` 虽然中途冲过阈值，最后分别回落到 `472.64` 和 `420.86`。
- 这意味着你最好的配置已经具备“达到解”的能力，但稳定性还不够，至少不是 `5/5` 稳定求解。
- 从跨 seed 波动也能看出来这一点：`lr=0.001` 的 seed 间标准差达到 `135.78`，说明随机种子影响仍然很大。

**训练动态怎么理解**
- 好的运行后期表现是健康的：我看的成功代表 run 里，后期 `ratio_mean` 很接近 `1.0`，`ratio_std` 约 `0.013~0.021`，`clip_fraction` 接近 `0`，`entropy` 下降到 `0.52~0.54` 左右，说明策略更新已经比较平稳，探索也在合理收敛。
- 差的低学习率运行更像“根本没真正学动”：后期 `entropy` 仍在 `0.677` 左右，`last_ma50` 只有 `24.46`，说明策略还接近高随机性状态。
- `T_horizon=256` 的失败也很典型：800 个 episode 里只做了 `3` 次 update，对应 summary 见 [summary.csv](file:///root/autodl-tmp/ppo/results/horizon_sensitivity/T_horizon_256/seed_0/summary.csv)。这不是简单的“学得差”，而是训练更新密度明显不够。
- 代表性地看，`T_horizon=256` 在最早几次更新里 `clip_fraction` 一度到 `0.15`、`ratio_std` 到 `0.068`，但总更新次数太少，后面根本没有足够机会修正策略。

**方法学上的注意点**
- 这套实验是按“完成多少个 episode”停止，而不是按“固定多少个环境步数”停止，代码见 [train.py:L83-L136](file:///root/autodl-tmp/ppo/train.py#L83-L136)。这会引入预算偏差。
- 直白说就是：表现更好的策略会活得更久，每个 episode 更长，因此在 `800` 个 episode 的预算下，它天然会获得更多环境步和更多参数更新。
- 所以 `updates` 更多不一定完全是“超参数更优”的结果，也部分是“强策略训练得更久”的结果；这会放大强配置和弱配置的差距。
- 这不影响你当前的经验性结论，但如果要写成正式实验结论，最好把训练预算改成固定 `timesteps` 再复现实验。

**最终结论**
- 当前实验最可靠的结论是：你的 PPO 实现是有效的，但默认设置还不够稳，真正把成绩拉起来的是更大的学习率 `0.001` 和更小的 rollout horizon `20`。
- `eps_clip` 在 `0.05~0.3` 范围内影响很小，不是当前优先调参方向。
- 最有价值的 trick 是 `advantage normalization`；`entropy bonus` 在这组实验里没有带来收益，反而明显拖慢/削弱了学习。
- `grad clip` 的实验结果和 baseline 完全一致，这一项目前不能下“有帮助”的结论，反而应该先检查它是否真的生效。
- 如果一句话概括：这批结果说明“PPO 能学会，但还没稳定学会；最优路线是提高学习率、保持短 horizon、保留 advantage normalization，并重新做固定 timesteps 的对比实验”。

**建议下一步**
- 优先围绕 `lr=0.001` 再细调，例如试 `7e-4`、`1e-3`、`1.2e-3`。
- 保持 `T_horizon=20` 或最多试更小的 `16/32`，不要再往 `64+` 走。
- 保留 `adv_norm`，重新评估 `entropy_coef`，很可能需要更小甚至关掉。
- 单独检查 `grad_clip` 是否真的触发，或者日志是否把 `baseline` 和 `grad_clip` 混成了同一组。
- 如果你愿意，我下一步可以直接把这些结果整理成一份“实验报告版”表格，或者继续把每个 `plot` 图逐张解释给你。


----
## 根据第一次微调，做第二次微调 ，提升稳定性
待做
