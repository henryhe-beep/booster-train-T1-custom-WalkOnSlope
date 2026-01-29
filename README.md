# Booster-T1-Walk-Custom-Slope

基于 NVIDIA Isaac Lab 的强化学习训练框架，专门为 Booster T1 全尺寸仿人机器人设计。

本仓库在 booster_train 的基础上进行了深度定制，将传统的运动模仿（BeyondMimic）框架修改为更具通用性的速度追踪（Velocity Tracking）框架，并引入了高度场斜坡地形（Slope Terrain），使机器人能够自主学习如何在平地及斜坡上行走。

---

## 🌟 主要特性

- **机器人型号**: Booster T1 全尺寸仿人机器人。
- **训练模式**: 速度追踪（Velocity Tracking）。无需 `.npz` 或 `.csv` 运动捕捉数据，机器人通过奖励函数自主探索行走步态。
- **地形支持**: 自定义高度场（Height Field）地形，包含平地、金字塔斜坡（Pyramid Slopes）以及随机粗糙地面。
- **课程学习**: 开启 Curriculum Learning，地形难度会随着机器人行走能力的提升而自动增加。
- **算法后端**: 使用高性能的 RSL_RL (PPO) 算法。

---

## 🛠️ 环境准备

在开始之前，请确保你已经安装并配置好以下环境：

1. **Isaac Sim & Isaac Lab**: 推荐使用最新版本。
2. **booster_assets**: 必须克隆并安装此资源库以获取 T1 的模型文件。

```bash
git clone https://github.com/BoosterRobotics/booster_assets.git
cd booster_assets && pip install -e .
```

---

## 🚀 快速开始

### 1. 安装本仓库

```bash
git clone https://github.com/henryhe-beep/booster-train-T1-custom-WalkOnSlope.git
cd booster-train-T1-custom-WalkOnSlope
pip install -e source/booster_train
```

### 2. 检查任务列表

运行以下脚本确认 T1 任务已成功注册：

```bash
python scripts/list_envs.py
```

你应该能看到 `Booster-T1-Walk-Custom-v0`。

### 3. 开始训练

使用以下命令启动 Headless 模式训练（推荐使用 2048 个并行环境以适配 RTX 4060 等显卡）：

```bash
python scripts/rsl_rl/train.py --task Booster-T1-Walk-Custom-v0 --num_envs 2048 --headless
```

### 4. 测试与可视化

查看训练好的模型效果（默认加载最新 Checkpoint）：

```bash
python scripts/rsl_rl/play.py --task Booster-T1-Walk-Custom-v0-Play
```

---

## 📂 项目结构说明

- **核心配置**: `source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/t1/walk_custom_slope/`
  - `tracking_env_cfg.py`: 定义了观测值、奖励函数（速度追踪、姿态维持）和终止条件。
  - `env_cfg.py`: 定义了 T1 机器人资产加载、PD 参数以及斜坡地形生成器。
  - `ppo_cfg.py`: RSL_RL 算法的超参数配置。
  - `__init__.py`: 任务 ID 注册。

---

## 📈 训练监控

你可以使用 TensorBoard 实时查看奖励曲线和地形等级：

```bash
tensorboard --logdir=logs/rsl_rl/t1_walk_custom_slope
```

---

## 🤝 致谢

感谢 Booster Robotics 开源的原始 booster_train 项目。