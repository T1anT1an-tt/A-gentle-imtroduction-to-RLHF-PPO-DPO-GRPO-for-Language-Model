# PPO 稳定性实验框架与扩展解析

本目录提供一套基于 PyTorch 和 Gymnasium 的 PPO 可复现实验框架，并配套整理了普通 PPO、连续动作 PPO 和 PPO-LSTM 的实现解析。代码在 MinimalRL PPO （https://github.com/T1anT1an-tt/minimalRL_Tt）实现的基础上进行了修正和扩展，重点用于观察 PPO 训练稳定性，而不是单纯追求最高分。

当前已经完成微调实验框架的是 **CartPole 离散动作 PPO**。连续动作 PPO 和 PPO-LSTM 目前主要以代码解析和实现对比的形式整理，尚未纳入同一套批量微调与稳定性实验流程。

## 内容概览

本目录可以分为两类内容：

- **实验框架代码**：用于运行 PPO 稳定性实验、保存日志并绘制结果。
- **学习解析文档**：用于解释普通 PPO、连续动作 PPO 和 PPO-LSTM 的实现差异。

## 实验框架代码

实验框架由 4 个核心 Python 文件组成。

### 1. `ppo.py`：核心网络与 PPO 算法

`ppo.py` 实现 PPO 的 Actor-Critic 结构，并包含以下改进：

- **固定 TD target 与 advantage**：将 TD target 和 advantage 的计算移到 `K_epoch` 循环外，避免它们随着 Critic 在同一批数据上的多轮更新而不断变化。
- **可开关的稳定化技巧**：
  - Advantage normalization：通过 `use_adv_norm` 控制。
  - Entropy bonus：通过 `use_entropy` 控制。
  - Gradient clipping：通过 `use_grad_clip` 控制。
- **训练指标记录**：`train_net()` 返回 `actor_loss`、`critic_loss`、`entropy`、`total_loss`、`ratio_mean`、`ratio_std`、`clip_fraction`、`adv_mean`、`adv_std` 等内部训练指标。


### 2. `train.py`：单次训练与日志记录

`train.py` 负责与 `CartPole-v1` 环境交互，并完成单次训练日志记录。

主要功能包括：

- 执行 rollout 与 episode 级训练循环。
- 记录每个 episode 的原始 return。
- 计算 moving average，包括窗口 20 和窗口 50。
- 记录 time to solve，即 moving average 首次达到 475 分所在的 episode。
- 将超参数配置、`episode_logs.csv` 和 `train_metrics.csv` 保存到对应结果目录。

### 3. `run_experiments.py`：批量实验与超参数扫描

`run_experiments.py` 用于批量运行 PPO 稳定性实验。默认会在 `seeds = [0, 1, 2, 3, 4]` 上重复实验，以观察不同随机种子下的表现差异。

实验分为 4 组：

1. **Learning rate sensitivity**：`lr = [1e-4, 3e-4, 5e-4, 1e-3]`
2. **Clip range sensitivity**：`eps_clip = [0.05, 0.1, 0.2, 0.3]`
3. **Rollout length sensitivity**：`T_horizon = [20, 64, 128, 256]`
4. **Stabilization ablation**：`baseline`、`adv_norm`、`entropy`、`grad_clip`、`all`


### 4. `plot_results.py`：实验结果可视化

`plot_results.py` 会扫描 `results/` 下的实验日志，并使用 `seaborn` 生成对比图。

默认输出包括：

- Mean reward curve，并显示不同 seed 之间的标准差阴影。
- Final 100 episode return boxplot。
- Policy entropy curve。
- Actor loss、critic loss 和 total loss 曲线。
- PPO ratio mean、ratio std 和 clip fraction 曲线。
- Advantage mean 曲线。

## 运行方式

当前可直接运行的是 CartPole 离散动作 PPO 稳定性实验。实验脚本依赖 `pandas`、`seaborn` 和 `matplotlib` 进行数据处理和绘图。可以使用 `uv` 临时安装依赖并运行脚本。

在当前目录 `ppo/` 下执行：

```powershell
uv run --with pandas --with seaborn --with matplotlib python run_experiments.py
```

该命令会运行所有参数组合和随机种子。默认设置下，实验规模约为：

```text
5 个 seed × 800 episodes × 多组超参数配置
```

实验完成后，执行可视化脚本：

```powershell
uv run --with pandas --with seaborn --with matplotlib python plot_results.py
```

生成的图像位于 `results/` 目录下各实验子目录的 `plots/` 文件夹中。

## 阅读实验结果时关注什么

分析结果时可以重点关注：

- reward 曲线是否平滑上升，还是出现明显震荡或回退。
- entropy 是否过早下降，提示策略可能过早变得确定。
- clip fraction 是否长期过高，提示策略更新可能经常被裁剪。
- ratio mean 和 ratio std 是否偏离正常范围。
- critic loss 是否长期震荡，提示价值函数拟合不稳定。
- 不同随机种子之间的方差是否明显扩大。

通过这些指标，可以更清楚地理解 PPO 中超参数和稳定化技巧对训练稳定性的影响。
