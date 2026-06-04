# PPO 稳定性实验框架 (PPO Stability Experiment Framework)

本目录包含一套基于 PyTorch 和 Gymnasium 的 PPO 可复现实验框架。该框架在原始 MinimalRL PPO 的基础上进行了 Bug 修复、功能扩展以及针对强化学习稳定性的优化。本框架的主要目的是为了**通过微调实验直观观察 PPO 的训练稳定性**，而非单纯追求最高分。

## 目录结构与功能拆解

整个实验框架被清晰地分为以下 4 个核心文件：

### 1. 核心网络与算法 `ppo.py`
实现了 PPO 的核心 Actor-Critic 架构，并做出了以下重要改进：
- **正确的 TD Target 计算**：修复了原版代码在 `K_epoch` 循环内部重复计算 Advantage 和 TD Target，导致其随着 Critic 网络更新而游移的 Bug。现在它们被抽离到了循环外部并固定。
- **三种稳定性 Tricks（支持开关）**：
  - **Advantage Normalization** (`use_adv_norm`)
  - **Entropy Bonus** (`use_entropy`)
  - **Gradient Clipping** (`use_grad_clip`)
- **丰富的指标收集**：单步 `train_net()` 会返回包括 `actor_loss`、`critic_loss`、`entropy`、`total_loss`、`ratio_mean`/`ratio_std`、`clip_fraction` 以及 `adv_mean`/`adv_std` 在内的全面内部状态指标。

### 2. 单次训练与日志记录 `train.py`
- 接管与 `CartPole-v1` 环境的交互（Rollout 过程）。
- 每个 Episode 计算并记录原始 reward 以及 moving average（窗口 20 和 50）。
- 追踪 **"Time to solve"**（即 moving average 首次达到 475 分所在的 episode）。
- 自动将单次训练的所有超参数配置、每个 episode 的日志 (`episode_logs.csv`) 以及训练内部指标 (`train_metrics.csv`) 保存到特定的结果目录下。

### 3. 多线程超参扫描 `run_experiments.py`
构建了针对 PPO 稳定性的四组核心消融/敏感度实验，自动遍历指定的 `seeds = [0, 1, 2, 3, 4]`：
1. **Learning rate sensitivity**: `lr = [1e-4, 3e-4, 5e-4, 1e-3]`
2. **Clip range sensitivity**: `eps_clip = [0.05, 0.1, 0.2, 0.3]`
3. **Rollout length sensitivity**: `T_horizon = [20, 64, 128, 256]`
4. **Stabilization ablation**: `baseline`, `adv_norm`, `entropy`, `grad_clip`, `all` (基线代表所有 Tricks 全关)。

实验结果默认保存在 `results/` 目录下，并最终聚合出一个 `summary_results.csv`。

### 4. 实验结果可视化 `plot_results.py`
自动扫描 `results/` 文件夹，使用 `seaborn` 绘制并保存科研级别的对比图：
- 每个参数设置的 Mean reward curve（带 seed 间标准差阴影）
- Final 100 episode return boxplot
- Policy entropy curve
- Actor / Critic / Total Loss curves
- PPO Ratio Mean/Std & Clip fraction curves
- Advantage Mean curves

图片会按照实验分组保存在各自目录的 `plots/` 文件夹中。

---

## 🚀 运行指南

由于框架引入了 `pandas`, `seaborn` 和 `matplotlib` 用于数据处理和画图，请确保使用 `uv` 运行或已在环境中安装这些依赖。

在当前目录 (`ppo/`) 下依次执行以下命令：

### 第一步：运行所有实验
运行这步需要一定的时间，因为它会自动跑完所有参数组合和随机种子（5个 seed × 800 episodes）：
```powershell
uv run --with pandas --with seaborn --with matplotlib python run_experiments.py
```

### 第二步：生成可视化图表
当上面的实验运行结束后，执行画图脚本提取所有的 CSV 数据并生成对比图：
```powershell
uv run --with pandas --with seaborn --with matplotlib python plot_results.py
```

执行完毕后，您可以在 `results/` 目录下各个实验子文件夹中的 `plots/` 里查看生成的 PNG 图像。通过这些图像，您可以直观地观察到不同超参数和 Tricks 对 PPO 训练过程（如早期的策略崩塌、Ratio 裁剪频率等）的深刻影响。
