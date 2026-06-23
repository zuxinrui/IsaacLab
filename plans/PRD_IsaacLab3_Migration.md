# PRD: LynxLab — Lynx 机械臂任务迁移至 IsaacLab 3.0 + Newton

> **版本**: v0.2
> **日期**: 2026-03-30
> **状态**: 待确认

---

## 1. 项目背景

### 1.1 现状问题

当前代码库（`~/IsaacLab`）基于 IsaacSim 4.5 → 5.1 逐步迁移，存在以下问题：

| 问题 | 详情 |
|------|------|
| **IsaacLab 源码被修改** | `simulation_context.py`（GPU interop fix）、`reward_manager.py`（SAC-X 多奖励支持）、`manager_based_rl_env.py`（多奖励观测组） |
| **版本混乱** | 从 IsaacSim 4.5 迁移到 5.1，中间累积了大量补丁 |
| **耦合严重** | 自定义任务代码散布在 IsaacLab 内部目录中，无法独立升级 |
| **Deformable workaround** | 为绕过 PhysX GridCloner 的 FEM bug，在 `simulation_context.py` 中硬编码修复 |

### 1.2 目标

搭建一套 **全新的、干净的** 开发环境：

- **IsaacSim 6.0** + **IsaacLab 3.0** + **Newton 物理后端**（先 PhysX 验证，后切 Newton）
- **零修改** IsaacLab 源码
- 所有 Lynx自定义工作以 **独立仓库 `LynxLab`** 形式存在
- 已有任务（reach / push / ball-in-cup / deformable push）全部可用
- 现有 `~/IsaacLab` 仓库 **完全不动**，仅重命名为 `~/IsaacLab-legacy`

---

## 2. 整体架构

### 2.1 三仓库结构

```
~/
├── IsaacLab-legacy/        # 当前仓库，重命名保留，不做任何修改
│                           # 含已训练模型 (logs/, outputs/)
│
├── IsaacLab/               # 全新 clone，IsaacLab 3.0 官方仓库
│                           # 零修改，纯净安装
│
└── LynxLab/               # 独立仓库，Lynx 机械臂所有自定义工作
                            # pip install -e . 安装到同一 conda 环境
```

### 2.2 Conda 环境

```
conda env: isaaclab3
├── IsaacSim 6.0            (pip install)
├── IsaacLab 3.0            (pip install -e ~/IsaacLab)
├── LynxLab                (pip install -e ~/LynxLab)
└── Newton physics backend  (随 IsaacSim 6.0 或单独安装)
```

---

## 3. 需要迁移的工作清单

### 3.1 自定义机器人资产

| 文件 | 大小 | 迁移 | 说明 |
|------|------|------|------|
| `lynx_constructor.py` | 54.7KB | **是** | 程序化 USD 构建器（基于 genotype 的模块化组装） |
| `lynx_ball_in_cup.py` | 13.2KB | **是** | 球杯变体（杯子+绳子+球的物理组装） |
| `lynx.py` | 3.6KB | **是** | 基础 Lynx配置入口 |
| `lynx_constructor_legacy.py` | 54.5KB | **丢弃** | 旧版构建器，不再需要 |

### 3.2 任务环境

| 任务 | 新注册名 | 关键文件 | 迁移 |
|------|---------|---------|------|
| **Reach (RL)** | `Lynx-Reach-v0~v5` | `reach_env_cfg.py` + `mdp/rewards.py` + Lynx configs | **是** |
| **Reach (Motion Planning)** | `Lynx-Reach-OMPL-v0` | 新增：基于 OMPL solver 的 reach | **新建** |
| **Push (Rigid)** | `Lynx-Push-Cube-v0` | `push_env_cfg.py` + `mdp/*` + config | **是** |
| **Push (Deformable)** | `Lynx-Push-DeformableCube-v0` | `joint_pos_deformable_env_cfg.py` + direct env | **是** |
| **Ball-in-Cup** | `Lynx-Ball-In-Cup-v0~v2` | `ball_in_cup_env_cfg.py` + `mdp/*` + config | **是** |
| ~~Classic Lynx Reach~~ | — | `classic/lynx_reach/` | **不迁移** |

### 3.3 RL 算法

| 模块 | 文件数 | 说明 |
|------|--------|------|
| **SAC-X** | 6 | 多意图奖励学习，用于 ball-in-cup |

### 3.4 训练与控制脚本

| 脚本 | 迁移 |
|------|------|
| `train.py` | 是 |
| `custom_on_policy_runner.py` | 是 |
| `control_push.py` | 是 |
| `control_ball_in_cup.py` | 是 |
| `control_deformable_push.py` | 是 |
| `readme.sh` | 是（更新命令） |
| `test/` | 选择性迁移 |

### 3.5 当前对 IsaacLab 源码的修改（需要消除）

| 修改文件 | 修改内容 | 迁移策略 |
|---------|---------|---------|
| `simulation_context.py` | 为 deformable body 禁用 GPU interop | Newton 可能已原生支持；否则通过 SimulationCfg 参数或 `__post_init__` hook |
| `reward_manager.py` | SAC-X 多奖励支持 | `MultiRewardWrapper` 或继承 `ManagerBasedRLEnv` 子类覆写 |
| `manager_based_rl_env.py` | 多奖励观测组 | 同上，Env Wrapper 处理 |

---

## 4. LynxLab 目录结构

```
~/LynxLab/
├── pyproject.toml                    # Python 包配置
├── README.md
├── .gitignore
│
├── lynx_lab/                        # 主 Python 包
│   ├── __init__.py
│   │
│   ├── assets/                       # 机器人资产
│   │   ├── __init__.py
│   │   ├── lynx.py                  # Lynx基础配置
│   │   ├── lynx_constructor.py      # 程序化 USD 构建器
│   │   └── lynx_ball_in_cup.py      # 球杯变体构建器
│   │
│   ├── tasks/                        # 任务环境
│   │   ├── __init__.py               # Gym 环境注册
│   │   │
│   │   ├── reach/                    # Reach 任务
│   │   │   ├── __init__.py
│   │   │   ├── reach_env_cfg.py
│   │   │   ├── mdp/
│   │   │   │   ├── __init__.py
│   │   │   │   └── rewards.py
│   │   │   └── config/
│   │   │       ├── rl/               # RL 学习的 reach
│   │   │       │   ├── joint_pos_env_cfg.py
│   │   │       │   ├── delta_pos_env_cfg.py
│   │   │       │   └── trapezoidal_env_cfg.py
│   │   │       └── motion_planning/  # OMPL solver reach
│   │   │           └── ompl_env_cfg.py
│   │   │
│   │   ├── push/                     # Push 任务
│   │   │   ├── __init__.py
│   │   │   ├── push_env_cfg.py
│   │   │   ├── mdp/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── rewards.py
│   │   │   │   ├── observations.py
│   │   │   │   └── terminations.py
│   │   │   └── config/
│   │   │       ├── rigid/            # 刚体 cube push
│   │   │       │   ├── joint_pos_env_cfg.py
│   │   │       │   └── obs_delay_env.py
│   │   │       └── deformable/       # 软体 cube push
│   │   │           ├── joint_pos_deformable_env_cfg.py
│   │   │           └── deformable_push_direct_env.py
│   │   │
│   │   └── ball_in_cup/              # Ball-in-Cup 任务
│   │       ├── __init__.py
│   │       ├── ball_in_cup_env_cfg.py
│   │       ├── mdp/
│   │       │   ├── __init__.py
│   │       │   ├── rewards.py
│   │       │   └── observations.py
│   │       └── config/
│   │           ├── v0_env_cfg.py
│   │           ├── v1_env_cfg.py     # with domain randomization
│   │           └── v2_env_cfg.py     # simplified
│   │
│   ├── rl/                           # 自定义 RL 算法
│   │   ├── __init__.py
│   │   └── sacx/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── models.py
│   │       ├── multi_reward_wrapper.py
│   │       ├── replay.py
│   │       └── scheduler.py
│   │
│   └── utils/                        # 工具函数
│       ├── __init__.py
│       └── env_wrappers.py           # MultiRewardWrapper 等
│
├── scripts/                          # 训练与控制脚本
│   ├── train.py
│   ├── custom_on_policy_runner.py
│   ├── control_push.py
│   ├── control_ball_in_cup.py
│   ├── control_deformable_push.py
│   ├── readme.sh
│   └── test/
│
└── outputs/                          # 训练输出（.gitignore）
```

**命名变化**：所有文件中 `lynx` → `links`（代码内部统一使用 `links` 命名）

---

## 5. Gym 环境注册

```python
# lynx_lab/tasks/__init__.py

# ── Reach ────────────────────────────────────────
"Lynx-Reach-v0"              # 基础 joint position control (RL)
"Lynx-Reach-v1"              # 精简观测 (RL)
"Lynx-Reach-v2"              # 精简+loop (RL)
"Lynx-Reach-v3"              # Delta control (RL)
"Lynx-Reach-v4"              # EE observation (RL)
"Lynx-Reach-v5"              # Delta + SAC reward (RL)
"Lynx-Reach-Trapezoidal-v0"  # 梯形速度控制 (RL)
"Lynx-Reach-OMPL-v0"         # OMPL Motion Planning solver

# ── Push ─────────────────────────────────────────
"Lynx-Push-Cube-v0"              # 刚体 cube push
"Lynx-Push-Cube-Play-v0"         # 刚体 play 模式
"Lynx-Push-Cube-ObsDelay-v0"     # 刚体 + 观测延迟
"Lynx-Push-DeformableCube-v0"    # 软体 cube push
"Lynx-Push-DeformableCube-Play-v0"

# ── Ball-in-Cup ──────────────────────────────────
"Lynx-Ball-In-Cup-v0"        # 基础
"Lynx-Ball-In-Cup-Play-v0"   # Play 模式
"Lynx-Ball-In-Cup-v1"        # + Domain Randomization
"Lynx-Ball-In-Cup-v2"        # 简化版
```

---

## 6. 技术方案

### 6.1 环境搭建流程

```bash
# Step 0: 保留现有仓库
mv ~/IsaacLab ~/IsaacLab-legacy

# Step 1: 新 conda 环境
conda create -n isaaclab3 python=3.11 -y
conda activate isaaclab3

# Step 2: 安装 IsaacSim 6.0
pip install isaacsim==6.0.0    # 或按官方文档

# Step 3: 安装 IsaacLab 3.0（官方仓库，零修改）
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
pip install -e .               # 或按官方 isaaclab.sh 脚本

# Step 4: 配置 Newton（后续，先用 PhysX 验证）
# pip install newton-physics    # 或随 IsaacSim 6.0 自带

# Step 5: 安装 LynxLab
cd ~/LynxLab
pip install -e .
```

### 6.2 消除 IsaacLab 源码修改

#### 6.2.1 GPU Interop Fix → SimulationCfg 配置

```python
# 在 LynxLab 的环境配置中，通过 SimulationCfg 设置
# 而非修改 IsaacLab 源码
class DeformablePushEnvCfg:
    sim: SimulationCfg = SimulationCfg(
        # IsaacLab 3.0 预计已暴露此选项
        use_fabric=False,  # 或 gpu_interop=False
    )
```

若 IsaacLab 3.0 未暴露该选项，在环境 `__post_init__` 中通过 Omniverse API 设置：

```python
def __post_init__(self):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    stage.SetMetadata("/physics/fabricUseGPUInterop", False)
```

#### 6.2.2 多奖励支持 → LynxLab 内部 Wrapper

```python
# lynx_lab/utils/env_wrappers.py
class MultiRewardEnvWrapper(gymnasium.Wrapper):
    """将多个 reward term 分组为 SAC-X 所需的多意图 reward signal"""

    def __init__(self, env, reward_groups: dict[str, list[str]]):
        super().__init__(env)
        self.reward_groups = reward_groups

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["reward_intentions"] = {
            group: sum(info["log"].get(t, 0) for t in terms)
            for group, terms in self.reward_groups.items()
        }
        return obs, reward, terminated, truncated, info
```

#### 6.2.3 Deformable GridCloner → Newton 原生 / 封装 Spawner

Newton 的核心卖点之一就是 deformable body 的高效多环境支持。预计迁移后此 workaround 可直接删除。

若仍需要，封装为 `lynx_lab/utils/deformable_spawner.py`。

### 6.3 Reach 任务两种模式

| 模式 | 说明 | 实现 |
|------|------|------|
| **RL** | 通过强化学习训练 policy 到达目标 | 现有 reach 任务迁移 |
| **Motion Planning** | 通过 OMPL solver 直接规划路径 | 新建：基于 IsaacLab 的 motion_gen 或 curobo 集成 |

Motion Planning 模式的 reach 需要在 Phase 3 中新建，可基于 IsaacLab 3.0 的 motion planning 工具或 cuRobo 集成。

### 6.4 Newton 物理后端（Phase 5 切换）

先在 PhysX 后端上完成所有迁移和验证，确认功能正确后再切换 Newton。

需要验证：
1. Newton 是否支持程序化 USD 构建的 Lynx机器人
2. Newton 的 deformable body 多环境支持
3. Newton 的 contact sensor / joint controller API 兼容性
4. Ball-in-cup 绳子物理行为一致性

---

## 7. 执行计划

### Phase 0: 环境准备（手动）
1. `mv ~/IsaacLab ~/IsaacLab-legacy`
2. 创建 conda 环境 `isaaclab3`
3. 安装 IsaacSim 6.0 + IsaacLab 3.0
4. 运行 IsaacLab 内置 demo 验证安装

### Phase 1: LynxLab 骨架搭建
1. 创建 `~/LynxLab/` 项目，初始化 git
2. 搭建目录结构 + `pyproject.toml`
3. `pip install -e .` 验证可导入
4. 文件命名统一：`lynx` → `links`

### Phase 2: 机器人资产迁移
1. 迁移 `lynx_constructor.py`、`lynx.py`、`lynx_ball_in_cup.py`
2. 更新 import 路径
3. 验证 Lynx机器人在新环境中正确构建和渲染

### Phase 3: 任务迁移（按优先级）
1. **Reach (RL)** — 最简单，作为验证基准
2. **Push (Rigid)** — 验证物体交互
3. **Ball-in-Cup** — 验证复杂物理（绳子/球/杯）
4. **Push (Deformable)** — 验证 deformable body
5. **Reach (Motion Planning)** — 新建 OMPL solver 版本

### Phase 4: RL 算法与脚本迁移
1. 迁移 SAC-X 模块
2. 迁移训练/控制脚本
3. 实现 `MultiRewardWrapper`
4. 端到端训练验证

### Phase 5: Newton 后端切换
1. 启用 Newton 物理后端
2. 逐个任务验证行为一致性
3. 性能基准测试（Newton vs PhysX）
4. 调参（如物理行为有差异）

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| IsaacSim 6.0 / IsaacLab 3.0 API 重大变更 | 先确认可用性，保留 `IsaacLab-legacy` 回退 |
| Newton 不支持程序化 USD 机器人 | Phase 0 做最小验证 |
| Newton deformable API 不兼容 | PhysX 作为 fallback |
| 多奖励支持仍需改源码 | Wrapper 方案兜底 |
| Ball-in-cup 绳子物理行为不同 | 先 PhysX 验证，后 Newton 调参 |

---

## 9. 决策记录

| # | 决策 | 结论 |
|---|------|------|
| 1 | 项目名 | **LynxLab**，独立仓库 |
| 2 | Legacy 构建器 | **丢弃** `lynx_constructor_legacy.py` |
| 3 | Classic Lynx Reach | **不迁移**；Reach 保留 RL + OMPL Motion Planning 两种 |
| 4 | 环境注册名 | **`Lynx-*`** 格式（如 `Lynx-Reach-v0`） |
| 5 | Newton 优先级 | **先 PhysX 验证迁移正确性，后切 Newton** |
| 6 | 已训练模型 | **保留在 `IsaacLab-legacy`**，不迁移 |
| 7 | 现有 IsaacLab 仓库 | **完全不动**，仅重命名为 `IsaacLab-legacy` |

---

## 10. 成功标准

- [ ] conda 环境 `isaaclab3`：IsaacSim 6.0 + IsaacLab 3.0 正常运行
- [ ] `LynxLab` 通过 `pip install -e .` 安装，**IsaacLab 源码零修改**
- [ ] 4 个任务（reach / push / ball-in-cup / deformable push）可注册和启动
- [ ] Reach 和 Push 任务可完成一轮训练并收敛
- [ ] Reach OMPL Motion Planning 模式可正常规划路径
- [ ] Newton 后端下仿真速度有明显提升
- [ ] `IsaacLab-legacy` 仓库完好无损
