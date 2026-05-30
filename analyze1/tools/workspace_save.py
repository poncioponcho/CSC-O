#!/usr/bin/env python3
"""
Workspace Save — 服务器端工作区保存工具
保存所有 tmux 会话、窗口布局、当前路径和运行命令
用法:
    python3 workspace_save.py              # 保存到默认位置
    python3 workspace_save.py /path/to/save.json  # 保存到指定位置
"""

import json
import os
import subprocess
import sys
from datetime import datetime


def run_cmd(cmd, check=False):
    """运行 shell 命令并返回输出"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def get_tmux_sessions():
    """获取所有 tmux 会话列表"""
    output = run_cmd("tmux list-sessions -F '#{session_name}|#{session_windows}|#{session_attached}'")
    sessions = []
    for line in output.split('\n'):
        if not line or '|' not in line:
            continue
        parts = line.split('|')
        sessions.append({
            "name": parts[0],
            "windows_count": int(parts[1]) if parts[1].isdigit() else 0,
            "attached": parts[2] == "1" if len(parts) > 2 else False
        })
    return sessions


def get_session_windows(session_name):
    """获取指定会话的所有窗口详情"""
    fmt = "#{window_index}|#{window_name}|#{pane_current_path}|#{pane_current_command}|#{pane_pid}"
    output = run_cmd(f"tmux list-windows -t '{session_name}' -F '{fmt}'")
    windows = []
    for line in output.split('\n'):
        if not line or '|' not in line:
            continue
        parts = line.split('|')
        windows.append({
            "index": int(parts[0]) if parts[0].isdigit() else 0,
            "name": parts[1],
            "path": parts[2],
            "command": parts[3],
            "pid": int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None
        })
    return windows


def get_session_panes(session_name):
    """获取指定会话的所有 pane 详情（用于精确恢复布局）"""
    fmt = "#{pane_id}|#{pane_current_path}|#{pane_current_command}|#{pane_pid}|#{pane_width}|#{pane_height}"
    output = run_cmd(f"tmux list-panes -t '{session_name}' -F '{fmt}'")
    panes = []
    for line in output.split('\n'):
        if not line or '|' not in line:
            continue
        parts = line.split('|')
        panes.append({
            "id": parts[0],
            "path": parts[1],
            "command": parts[2],
            "pid": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
            "width": int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 80,
            "height": int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 24,
        })
    return panes


def get_running_processes():
    """获取用户运行中的关键进程"""
    output = run_cmd("ps -u $(whoami) -o pid,ppid,pcpu,pmem,args --no-headers | grep -E 'python|tmux|jupyter' | grep -v grep")
    processes = []
    for line in output.split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            processes.append({
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "cpu": parts[2],
                "mem": parts[3],
                "command": " ".join(parts[4:])
            })
    return processes


def save_workspace(save_file=None):
    """保存完整工作区状态"""
    save_dir = os.path.expanduser("~/.vpn_workspace_saves")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save_file is None:
        save_file = os.path.join(save_dir, f"save_{timestamp}.json")

    state = {
        "version": "1.0",
        "timestamp": timestamp,
        "hostname": os.uname().nodename,
        "user": os.getenv("USER", "unknown"),
        "current_dir": os.getcwd(),
        "environment": {
            "CONDA_DEFAULT_ENV": os.getenv("CONDA_DEFAULT_ENV", ""),
            "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV", ""),
        },
        "sessions": [],
        "processes": get_running_processes()
    }

    for session in get_tmux_sessions():
        session_data = {
            "name": session["name"],
            "attached": session["attached"],
            "windows": get_session_windows(session["name"]),
            "panes": get_session_panes(session["name"])
        }
        state["sessions"].append(session_data)

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    # 更新 latest 软链接
    latest_link = os.path.join(save_dir, "latest.json")
    if os.path.islink(latest_link):
        os.remove(latest_link)
    elif os.path.exists(latest_link):
        os.replace(latest_link, latest_link + ".bak")
    os.symlink(save_file, latest_link)

    print(f"[✓] 工作区已保存: {save_file}")
    print(f"    会话数: {len(state['sessions'])}")
    print(f"    进程数: {len(state['processes'])}")
    return save_file


if __name__ == "__main__":
    save_file = sys.argv[1] if len(sys.argv) > 1 else None
    save_workspace(save_file)
