# A2C训练监控指南

## 概述

为了更好地判断A2C模型的收敛情况，我们增强了训练过程中的指标记录和可视化功能。现在系统会记录两类数据：

1. **Episode级别的环境交互指标** - 每个episode结束时记录
2. **A2C训练指标** - 每次rollout结束后记录（每n_steps一次）

---

## 📊 记录的指标

### 1. Episode级别指标（按episode索引）

**文件位置**: `results/logs/a2c/<run_name>/by_episode_reward.txt`

**格式**:
```
# episode reward feasible_rate success_rate mean_miou mean_latency episode_length
1 43.560000 0.760000 0.760000 0.633800 143.501500 100
2 42.840000 0.700000 0.700000 0.684800 158.833140 100
...
```

**字段说明**:
- `episode`: Episode索引（从1开始）
- `reward`: Episode总奖励
- `feasible_rate`: 可行步骤占比（满足延迟约束的步骤比例）
- `success_rate`: 成功率（当前等同于feasible_rate）
- `mean_miou`: Episode内平均mIoU（分割质量指标）
- `mean_latency`: Episode内平均总延迟（秒）
- `episode_length`: Episode长度（步数）

**TensorBoard标签** (横轴为episode索引):
- `by_episode/episode_reward`
- `by_episode/episode_length`
- `by_episode/feasible_rate`
- `by_episode/success_rate`
- `by_episode/mean_miou`
- `by_episode/mean_latency`

---

### 2. A2C训练指标（按timesteps）

**文件位置**: `results/logs/a2c/<run_name>/a2c_training_metrics.txt`

**格式**:
```
# timesteps n_updates policy_loss value_loss entropy_loss explained_variance learning_rate total_loss
32768 1 0.123456 0.234567 0.012345 0.567890 0.000300 0.370368
65536 2 0.112345 0.223456 0.011234 0.678901 0.000300 0.346035
...
```

**字段说明**:
- `timesteps`: 当前训练步数
- `n_updates`: 策略更新次数
- `policy_loss`: 策略损失（越小越好，但不应为0）
- `value_loss`: 价值函数损失（应逐渐下降并趋稳）
- `entropy_loss`: 熵损失（鼓励探索，应逐渐下降但保持非零）
- `explained_variance`: 解释方差（越接近1越好，表示价值函数拟合质量）
- `learning_rate`: 当前学习率
- `total_loss`: 总损失（policy_loss + vf_coef * value_loss - ent_coef * entropy_loss）

**TensorBoard标签** (横轴为timesteps):
- `train/policy_loss` (SB3原生)
- `train/value_loss` (SB3原生)
- `train/entropy_loss` (SB3原生)
- `train/explained_variance` (SB3原生)
- `train/learning_rate` (SB3原生)
- `train/n_updates` (SB3原生)
- `train/loss` (SB3原生)
- `a2c/policy_loss` (我们的副本，便于分组查看)
- `a2c/value_loss`
- `a2c/entropy_loss`
- `a2c/explained_variance`
- `a2c/learning_rate`
- `a2c/n_updates`
- `a2c/total_loss`

---

### 3. 实时训练指标（按timesteps）

**TensorBoard标签**:
- `train/step_reward_inst`: 当前步的即时奖励（VecEnv多环境平均）
- `train/step_reward_ma`: 滑动窗口平均奖励（默认窗口1000步）
- `train/step_reward_ema`: 指数移动平均奖励（更平滑的趋势）

---

## 🔍 如何判断A2C收敛

### 关键指标及其期望趋势

#### 1. **策略侧指标**

| 指标 | 期望趋势 | 说明 |
|------|---------|------|
| `policy_loss` | 下降后趋稳 | 策略梯度损失，应该逐渐减小并稳定在较低值 |
| `entropy_loss` | 逐渐下降但保持非零 | 熵损失，太低会导致过早收敛到次优策略 |
| `explained_variance` | 趋近1.0 | 价值函数对回报的解释能力，越接近1越好 |

#### 2. **价值侧指标**

| 指标 | 期望趋势 | 说明 |
|------|---------|------|
| `value_loss` | 下降并趋稳 | 价值函数拟合误差，应持续下降 |

#### 3. **环境交互指标**

| 指标 | 期望趋势 | 说明 |
|------|---------|------|
| `episode_reward` | 上升并趋稳 | Episode总奖励，应持续增长 |
| `feasible_rate` | 上升并趋稳 | 可行解比例，越高越好 |
| `mean_miou` | 上升（如果越高越好） | 分割质量，取决于任务目标 |
| `mean_latency` | 下降（如果越低越好） | 平均延迟，取决于优化目标 |
| `step_reward_ma` | 上升并趋稳 | 平滑的奖励趋势 |

#### 4. **训练动态指标**

| 指标 | 说明 |
|------|------|
| `learning_rate` | 学习率（如使用lr_schedule会变化） |
| `n_updates` | 策略更新次数，应线性增长 |

---

## 📈 使用TensorBoard查看

### 启动TensorBoard

```bash
tensorboard --logdir results/logs/a2c
```

然后在浏览器中打开 `http://localhost:6006`

### 推荐的面板配置

#### 面板1: 训练损失
- `train/policy_loss`
- `train/value_loss`
- `train/entropy_loss`
- `train/loss`

#### 面板2: 训练质量
- `train/explained_variance`
- `a2c/explained_variance`

#### 面板3: Episode性能
- `by_episode/episode_reward`
- `by_episode/feasible_rate`
- `by_episode/success_rate`

#### 面板4: 实时奖励
- `train/step_reward_inst`
- `train/step_reward_ma`
- `train/step_reward_ema`

#### 面板5: 任务指标
- `by_episode/mean_miou`
- `by_episode/mean_latency`

---

## 🚨 收敛问题诊断

### 问题1: 奖励不增长
**可能原因**:
- `explained_variance` < 0.5 → 价值函数拟合不好
- `entropy_loss` 过低 → 探索不足
- `policy_loss` 震荡 → 学习率过高

**解决方案**:
- 降低学习率
- 增加 `ent_coef`（熵系数）
- 检查环境奖励设计

### 问题2: 训练不稳定
**可能原因**:
- `value_loss` 震荡剧烈
- `policy_loss` 突然增大

**解决方案**:
- 降低学习率
- 减小 `n_steps`（更频繁更新）
- 增加 `vf_coef`（价值函数权重）

### 问题3: 过早收敛
**可能原因**:
- `entropy_loss` 过快降至接近0
- `feasible_rate` 停滞在较低水平

**解决方案**:
- 增加 `ent_coef`
- 使用学习率衰减
- 检查奖励函数是否有局部最优

---

## 📁 文件结构

训练运行后，会生成以下文件结构：

```
results/logs/a2c/<run_name>/
├── by_episode_reward.txt          # Episode级别指标
├── a2c_training_metrics.txt       # A2C训练指标
├── tb/                             # TensorBoard日志
│   └── A2C_1/
│       └── events.out.tfevents.*
├── monitor/                        # Monitor日志（SB3原生）
│   ├── monitor_0.csv
│   ├── monitor_1.csv
│   └── ...
├── train_logs/                     # 详细训练日志（JSONL格式）
│   ├── train_ep_000001-000100.jsonl
│   └── ...
└── train_plain_logs/               # 人类可读训练日志
    ├── train_ep_000001-000100.log
    └── ...
```

---

## 🔧 自定义配置

### 修改记录频率

在 `scripts/train_a2c.py` 中调用 `build_callbacks` 时：

```python
callbacks = build_callbacks(
    ...
    a2c_metrics_log_every=100,  # A2C指标记录频率（每100步）
    step_ma_log_every=200,      # 步级奖励记录频率
    ...
)
```

### 禁用某些记录

```python
callbacks = build_callbacks(
    ...
    enable_a2c_metrics=False,        # 禁用A2C指标记录
    enable_episode_reward=False,     # 禁用Episode指标记录
    enable_step_reward_ma=False,     # 禁用步级奖励记录
    ...
)
```

---

## 💡 最佳实践

1. **训练初期**（前10-20%步数）:
   - 重点关注 `explained_variance` 是否快速上升
   - 检查 `value_loss` 是否下降
   - 确保 `entropy_loss` 保持在合理范围（不要太快降至0）

2. **训练中期**（20-70%步数）:
   - 关注 `episode_reward` 是否持续增长
   - 检查 `feasible_rate` 是否提升
   - 监控 `policy_loss` 是否趋稳

3. **训练后期**（70-100%步数）:
   - 确认各指标是否收敛（曲线趋于平缓）
   - 检查 `explained_variance` 是否接近1.0
   - 评估最终性能是否满足要求

4. **定期检查**:
   - 每隔一段时间查看TensorBoard
   - 对比不同运行的曲线
   - 保存表现好的checkpoint

---

## 📚 参考资料

- [Stable-Baselines3 A2C文档](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
- [A2C算法论文](https://arxiv.org/abs/1602.01783)
- [TensorBoard使用指南](https://www.tensorflow.org/tensorboard)

---

**更新日期**: 2025-12-19
**版本**: 1.0

