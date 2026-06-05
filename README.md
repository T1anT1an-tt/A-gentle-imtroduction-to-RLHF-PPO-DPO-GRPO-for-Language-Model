# PPO、DPO 与 GRPO 在 RLHF 中的应用学习笔记

本仓库整理了围绕 RLHF（Reinforcement Learning from Human Feedback）中常见优化方法的学习笔记、公式推导和 PPO 实验代码。内容主要来自对 PPO、DPO、GRPO 及相关强化学习概念的系统梳理

视频链接：[浅析大模型后训练PPO_DPO_GRPO等强化学习算法（上）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1okG16sEiW?spm_id_from=333.788.videopod.sections&vd_source=1ca710c6c85307e10bd1bfbc2dd45e76)

当前仓库的重点包括：

- DPO 推导中从 RLHF KL 约束目标到偏好优化损失的关键步骤
- 神经网络结构限制对理论最优策略 $\pi^*$ 与实际策略 $\pi_\theta$ 的影响
- PPO 经典实现、稳定性实验框架和可视化分析
- 连续动作空间 PPO 与 PPO-LSTM 的实现差异解析
- RLHF 中 PPO、DPO、GRPO 等方法之间的关系梳理

## 项目目标

回答学习 RLHF 算法时常见的几个问题：

1. PPO 为什么会成为早期 RLHF 中常用的策略优化方法？
2. DPO 为什么可以绕过显式奖励模型和在线强化学习过程？
3. 连续动作、离散动作、序列记忆等实现细节会如何改变 PPO 的代码结构？
4. 理论推导中的最优策略与真实神经网络训练之间有什么差距？
5. 面向大语言模型对齐时，PPO、DPO、GRPO 分别解决了什么问题？

## 适合读者

本仓库适合以下读者：

- 正在学习 RLHF、PPO、DPO 或 GRPO 的读者。
- 已经了解基础强化学习，但希望把公式推导和代码实现对应起来的读者。
- 想通过 PyTorch/Gymnasium 代码观察 PPO 训练稳定性的读者。
- 希望从普通 PPO 进一步理解连续动作 PPO、PPO-LSTM 等变体的读者。

阅读前建议具备以下基础：

- 基本的强化学习概念，例如 policy、reward、value function、advantage。
- 基本的 PyTorch 神经网络训练流程。
- 对 KL divergence、log probability、softmax、Normal distribution 有初步了解。

## 仓库结构

```text
.
├── DPO省略版推导.md
├── 神经网络结构的限制是什么.md
├── ppo/
│   ├── README.md
│   ├── ppo.py
│   ├── train.py
│   ├── run_experiments.py
│   ├── plot_results.py
│   ├── PPO-continuous.md
│   ├── PPO-LSTM.md
│   ├── LSTM训练细节附件.md
│   └── PPO算法经典实现及解析.pdf
└── 浅分析PPO等·算法 在RLHF中的应用.canvas
```

其中：

- `DPO省略版推导.md`：整理 DPO 从 RLHF KL 约束目标到 DPO loss 的核心推导。
- `神经网络结构的限制是什么.md`：解释理论最优策略与真实神经网络策略之间的差距。
- `ppo/`：包含 PPO 实验代码、结果可视化脚本和 PPO 变体解析文档。
- `浅分析PPO等·算法 在RLHF中的应用.canvas`：Obsidian Canvas 形式的本地知识图谱，用于辅助个人梳理；GitHub 网页端不一定适合直接阅读。

## 推荐阅读顺序

如果目标是理解 RLHF 中 PPO 和 DPO 的关系，可以按以下顺序阅读：

1. [神经网络结构的限制是什么.md](神经网络结构的限制是什么.md)
2. [DPO省略版推导.md](DPO省略版推导.md)
3. [ppo/README.md](ppo/README.md)
4. [ppo/PPO算法经典实现及解析.pdf](ppo/PPO算法经典实现及解析.pdf)
5. [ppo/PPO-continuous.md](ppo/PPO-continuous.md)
6. [ppo/PPO-LSTM.md](ppo/PPO-LSTM.md)
7. [ppo/LSTM训练细节附件.md](ppo/LSTM训练细节附件.md)

如果目标是运行 PPO 实验，可以直接从 [ppo/README.md](ppo/README.md) 开始。

## PPO 实验框架

`ppo/` 目录中提供了一个基于 PyTorch 和 Gymnasium 的 PPO 稳定性实验框架。当前已完成微调实验的是 `CartPole-v1` 离散动作 PPO，主要用于观察不同超参数和稳定化技巧对训练过程的影响。

实验覆盖：

- learning rate sensitivity
- clip range sensitivity
- rollout length sensitivity
- advantage normalization、entropy bonus、gradient clipping 的消融实验

运行入口：

```powershell
cd ppo
uv run --with pandas --with seaborn --with matplotlib python run_experiments.py
uv run --with pandas --with seaborn --with matplotlib python plot_results.py
```

## 关于图片和 PDF

仓库当前优先使用 Markdown 和 PDF 作为公开阅读材料。Obsidian 本地图片附件目录 `Attachments/` 已被 `.gitignore` 忽略，不作为仓库发布内容。

## 说明

本仓库是个人学习和整理型项目，使用的笔记软件是Obsidian，重点在于把 RLHF 相关算法的推导、代码和直觉解释对应起来。内容会随着学习过程持续修订，欢迎基于公式、实现细节或表达清晰度提出建议。
