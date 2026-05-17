<<<<<<< HEAD
# honor-ai

GitHub 仓库名：[hero-honor-ai](https://github.com/justicbro/hero-honor-ai)。

王者荣耀 AI 实验项目。当前阶段：**最小可行链路 demo**（Mac 模拟器采集 → 服务器推理 → 回传动作）。

> ⚠️ **风险声明**：王者荣耀有 TP 反作弊。请使用小号、训练营/人机模式进行实验，本项目仅供技术学习。

## 架构

```
┌──── 你的 Mac (Apple Silicon) ─────┐         ┌──── 你的服务器 ─────┐
│  Android 模拟器 (MuMu)            │         │                       │
│        ↕ ADB                       │         │                       │
│  mac_agent/agent.py  ─────────────────────► server/server.py        │
│      截屏 + 注入控制              │ WebSocket │   解码 + 推理       │
│                       ◄─────────────────────                         │
│                                    │  动作JSON│                       │
└────────────────────────────────────┘         └───────────────────────┘
```

- 上行：JPEG 二进制帧
- 下行：`Action` JSON（tap / swipe / noop）
- 协议见 `shared/protocol.py`

## 目录结构

```
honor-ai/
├── shared/protocol.py         # 两端共用的消息定义
├── mac_agent/                 # 跑在你 Mac 上
│   ├── agent.py               # 主循环
│   ├── capture.py             # ADB 截屏
│   ├── control.py             # ADB 输入注入
│   └── requirements.txt
├── server/                    # 跑在你的服务器上
│   ├── server.py              # WebSocket 服务端
│   ├── inference.py           # 决策（起步=模板匹配）
│   ├── templates/             # 放 tap_*.png 模板
│   └── requirements.txt
└── tests/mock_client.py       # 无 Mac 也能跑的链路自测
```

## 快速开始

### 第一步：服务器侧

```bash
cd /home/peilin/projects/honor-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
PYTHONPATH=. python -u -m server.server --host 0.0.0.0 --port 8765
```

开放服务器 8765 端口（防火墙/安全组）。

> 注意：必须用 `python -m server.server` 而不是 `python server/server.py`，否则会因 `server/` 同名包导致 `ModuleNotFoundError`。`-u` 让日志不缓冲。

### 第二步：链路自测（可在服务器本机或任意机器跑）

```bash
PYTHONPATH=. python -m tests.mock_client --server ws://127.0.0.1:8765
```

应看到一串 `rtt=XX ms action=Action(type='noop', ...)` 输出。`noop` 是因为还没有模板，链路通了。

### 第三步：Mac 侧

```bash
# 装 ADB
brew install android-platform-tools

# 启动 MuMu 模拟器，连 ADB（端口看你的模拟器：MuMu=7555, BlueStacks=5555）
adb connect 127.0.0.1:7555
adb devices   # 确认有设备

# 装依赖
cd ~/honor-ai   # 假设你 scp 了项目过来
pip install -r mac_agent/requirements.txt

# 启动 agent
PYTHONPATH=. python -m mac_agent.agent --server ws://YOUR_SERVER_IP:8765 --fps 5
```

### 第四步：放第一个模板

1. 进训练营到出现"开始"按钮的画面
2. Mac 上 `adb exec-out screencap -p > /tmp/full.png`
3. 用 macOS Preview 裁出按钮，存为 `tap_start.png`
4. `scp tap_start.png YOUR_SERVER:/home/peilin/projects/honor-ai/server/templates/`
5. 重启 server，看到 `loaded 1 templates: ['tap_start']`，agent 检测到该按钮就会自动点

## 训练营"无脑平A"最小 Demo（跳过模板匹配）

平A / 技能按钮在屏幕上位置是固定的，不需要 AI 看图。最快验证整条链路的方式：

### 1. 服务器侧用固定坐标 decider 启动

```bash
# 屏幕右下角 92% / 85% 处 = 平A 按钮（横屏王者荣耀通用位置）
# 每 5 帧 (fps=5 时即每秒) 发一次 tap
PYTHONPATH=. python -u -m server.server \
    --host 0.0.0.0 --port 8765 \
    --demo-tap 0.92,0.85
```

`--demo-tap` 接受三种格式：
- `0.92,0.85`     —— 相对坐标（推荐，跨分辨率通用）
- `1180,600`      —— 绝对像素
- `0.92,0.85@3`   —— 每 3 帧 tap 一次

启动后服务器会打印 `FixedTapDecider relative (0.92,0.85) every 5 frames`，等 Mac 连上后 hello 消息一到立刻打印 `resolved tap target -> (1177,612) on 1280x720`。

### 2. Mac 侧先 dry-run 验证

进训练营、英雄走到木桩附近，然后：

```bash
# 不真的点，只打印 server 返回的动作，看看坐标对不对
PYTHONPATH=. python -m mac_agent.agent \
    --server ws://YOUR_SERVER_IP:8765 \
    --fps 5 --dry-run --save-frame /tmp/first.jpg

# scp 出 /tmp/first.jpg 到本地用 Preview 看
# 确认服务器算出的 (1177, 612) 是不是落在了平A 按钮上
```

终端会刷出：

```
[agent] frame#4 -> Action(type='tap', x=1177, y=612, duration_ms=60) [skipped]
```

### 3. 去掉 --dry-run，英雄就开始自己平 A

```bash
PYTHONPATH=. python -m mac_agent.agent \
    --server ws://YOUR_SERVER_IP:8765 --fps 5
```

整条链路：模拟器截屏 → JPEG → 服务器 → tap JSON → ADB `input tap` → 模拟器收到点击，英雄出招。

### 想换技能？只改坐标，不改代码

横屏王者荣耀按钮的经验坐标（相对值）：

| 按钮 | --demo-tap |
|---|---|
| 平 A（基础攻击） | `0.92,0.85` |
| 技能 1（一） | `0.78,0.85` |
| 技能 2（二） | `0.83,0.70` |
| 技能 3（三 / 大招） | `0.92,0.55` |
| 回城 | `0.05,0.90` |
| 恢复 | `0.10,0.90` |

实际位置因模拟器分辨率和 UI 缩放略有差异。**用 `--save-frame` 抓一张实图，在 Preview 里量像素坐标除以分辨率**，得到的就是你自己的精确比例。

## 升级路线

| 阶段 | 改造 | 收益 |
|---|---|---|
| v0.2 | 采集换 `scrcpy-server` 走 H.264 | 30fps，端到端 < 100ms |
| v0.3 | YOLOv8 识别英雄/兵线/塔/小地图 | 真正"看懂"画面 |
| v0.4 | 控制换 `minitouch` 二进制协议 | 注入 50ms → 5ms |
| v0.5 | 决策接 PPO 强化学习 | 自学走位 / 连招 |
| v1.0 | ZeroMQ + 对局状态机 | 工程化 |

## 调参参考

- `--fps 5`：demo 阶段建议；ADB screencap 最多 ~5fps
- `--jpeg-quality 70`：质量/带宽平衡点；高分屏可降到 50
