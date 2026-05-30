# VPN 断连自动保存与恢复系统

针对校园 VPN 多用户共享环境下的间歇性断连问题，本系统提供完整的监测、保存、恢复能力。

## 系统架构

```
┌─────────────────┐         VPN隧道          ┌─────────────────┐
│   Mac 本地端     │  ═══════════════════════► │   服务器端       │
│                 │        (utun4)           │                 │
│ vpn_watchdog.sh │                         │ workspace_save.py │
│  · 延迟趋势预测  │                         │  · 保存tmux会话   │
│  · 断连检测     │                         │  · 保存进程状态   │
│  · 自动恢复SSH  │                         │ vim_session_save  │
│                 │                         │  · 保存vim编辑状态│
└─────────────────┘                         └─────────────────┘
```

## 组件说明

### Mac 端（本地）

| 脚本 | 功能 |
|------|------|
| `vpn_watchdog.sh` | **核心守护进程**。每 5 秒监测 VPN 状态，基于延迟趋势预测断连，断连时保存本地工作区，恢复时自动打开 Terminal + SSH + tmux attach |
| `ssh_auto_attach.sh` | 一键手动恢复：SSH 到服务器 + 恢复工作区 + attach tmux |
| `check_server_status.sh` | 一键查询服务器运行状态（GPU、管线进度、日志、磁盘） |

### 服务器端（远程）

| 脚本 | 功能 |
|------|------|
| `workspace_save.py` | 保存所有 tmux 会话、窗口布局、当前路径、运行进程到 JSON |
| `workspace_restore.py` | 从 JSON 快照恢复 tmux 会话和窗口 |
| `vim_session_save.sh` | 扫描 tmux pane 中的 vim/nano 进程，保存编辑文件列表 |
| `vim_session_restore.sh` | 在恢复的 tmux 会话中重新打开之前编辑的文件 |

### Pipeline 集成

`csco_pipeline.py` 的 `ProgressTracker` 已增强：
- 每个 stage 完成/出错时，在 `work/notifications/` 下写入状态文件
- VPN 恢复后可通过 `cat work/notifications/LATEST_STATUS.txt` 查看最新进度

## 快速开始

### 1. 启动 VPN 监测守护进程（Mac）

```bash
cd ~/Desktop/课题组/蛋白质结构序列实验_结构方向优化/analyze1/tools
chmod +x *.sh

# 前台运行（调试用）
./vpn_watchdog.sh

# 后台运行（推荐）
nohup ./vpn_watchdog.sh > ~/.vpn_watchdog.log 2>&1 &

# 查看日志
tail -f ~/.vpn_watchdog.log

# 停止
kill $(cat ~/.vpn_watchdog.pid)
```

### 2. 服务器端安装（只需一次）

```bash
# SSH 到服务器后
cd ~/CSC-O/tools
chmod +x *.sh

# 测试工作区保存
python3 workspace_save.py

# 测试工作区恢复
python3 workspace_restore.py
```

### 3. 日常使用场景

#### 场景 A：VPN 突然断开
1. `vpn_watchdog` 检测到断连 → 记录日志并发送 Mac 通知
2. 服务器上的 `tmux` 会话继续运行，`csco_pipeline --resume` 保证进度不丢失
3. ProgressTracker 的通知文件持续更新

#### 场景 B：VPN 恢复后自动恢复
1. `vpn_watchdog` 检测到 VPN 恢复 → 等待 3 秒路由稳定
2. 自动打开 Terminal → SSH → 执行 `workspace_restore.py` → `tmux attach`
3. 你直接看到断连前的工作界面

#### 场景 C：手动查询服务器状态（不启动 watchdog）
```bash
./check_server_status.sh
```

## 预测机制说明

`vpn_watchdog.sh` 不只是被动检测断连，还具备**预测能力**：

1. **延迟趋势窗口**：维护最近 20 次 ping 的延迟历史
2. **陡增检测**：如果最新延迟 > 平均值 × 3 且超过 300ms，触发预警
3. **波动检测**：如果标准差 > 平均值 × 0.8，判定网络不稳定
4. **丢包预警**：丢包率 ≥ 20% 时提前保存工作区

## 环境变量配置

可在 `~/.bashrc` 或 `~/.zshrc` 中预设：

```bash
export VPNWD_SERVER_IP="49.52.29.0"
export VPNWD_SERVER_USER="test"
export VPNWD_SESSION="csco"
export VPNWD_INTERVAL="5"
```

## 文件说明

```
tools/
├── vpn_watchdog.sh          # Mac 端 VPN 监测守护进程（核心）
├── ssh_auto_attach.sh       # Mac 端手动一键恢复
├── check_server_status.sh   # Mac 端服务器状态查询
├── workspace_save.py        # 服务器端工作区保存
├── workspace_restore.py     # 服务器端工作区恢复
├── vim_session_save.sh      # 服务器端 vim 状态保存
├── vim_session_restore.sh   # 服务器端 vim 状态恢复
└── README.md                # 本文件
```

## 注意事项

1. **tmux 会话名**：默认使用 `csco`，可在环境变量 `VPNWD_SESSION` 中修改
2. **osascript**：`vpn_watchdog` 使用 AppleScript 打开 Terminal，仅支持 macOS
3. **Pipeline 进度**：即使 VPN 断开，服务器上的 `csco_pipeline.py --resume` 仍会断点续跑
4. **敏感信息**：`vpn_watchdog` 日志保存在 `~/.vpn_watchdog.log`，不含密码
