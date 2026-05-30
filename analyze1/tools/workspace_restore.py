#!/usr/bin/env python3
"""
Workspace Restore — 服务器端工作区恢复工具
从保存的快照中恢复 tmux 会话和窗口布局
用法:
    python3 workspace_restore.py              # 从最新快照恢复
    python3 workspace_restore.py /path/to/save.json  # 从指定文件恢复
"""

import json
import os
import subprocess
import sys


def run_cmd(cmd, check=False):
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return f"ERROR: {e}", 1


def session_exists(name):
    """检查 tmux 会话是否已存在"""
    _, code = run_cmd(f"tmux has-session -t '{name}' 2>/dev/null")
    return code == 0


def restore_workspace(save_file=None):
    """从快照恢复工作区"""
    if save_file is None:
        save_file = os.path.expanduser("~/.vpn_workspace_saves/latest.json")

    if not os.path.exists(save_file):
        print(f"[✗] 未找到保存文件: {save_file}")
        print("    请先运行: python3 workspace_save.py")
        return False

    with open(save_file, 'r', encoding='utf-8') as f:
        state = json.load(f)

    print("========================================")
    print(f"恢复工作区: {state.get('timestamp', 'unknown')}")
    print(f"来源主机: {state.get('hostname', 'unknown')}")
    print(f"用户: {state.get('user', 'unknown')}")
    print("========================================\n")

    restored = 0
    skipped = 0

    for session in state.get("sessions", []):
        session_name = session["name"]

        if session_exists(session_name):
            print(f"[~] 会话 '{session_name}' 已存在，跳过")
            skipped += 1
            continue

        windows = session.get("windows", [])
        if not windows:
            print(f"[!] 会话 '{session_name}' 无窗口记录，跳过")
            continue

        # 创建第一个窗口（tmux 要求至少一个窗口才能创建会话）
        first = windows[0]
        start_path = first.get("path", os.path.expanduser("~"))
        start_cmd = first.get("command", "bash")

        # 创建会话
        run_cmd(
            f"tmux new-session -d -s '{session_name}' -c '{start_path}'",
            check=True
        )
        run_cmd(
            f"tmux rename-window -t '{session_name}:0' '{first.get('name', 'main')}'",
            check=False
        )

        # 如果原始命令不是 shell，尝试恢复
        if start_cmd not in ("bash", "sh", "zsh", ""):
            run_cmd(
                f"tmux send-keys -t '{session_name}:0' 'cd {start_path} && {start_cmd}' C-m",
                check=False
            )

        # 创建剩余窗口
        for window in windows[1:]:
            win_path = window.get("path", os.path.expanduser("~"))
            win_name = window.get("name", "window")
            win_cmd = window.get("command", "bash")

            run_cmd(
                f"tmux new-window -t '{session_name}' -c '{win_path}' -n '{win_name}'",
                check=True
            )

            if win_cmd not in ("bash", "sh", "zsh", ""):
                run_cmd(
                    f"tmux send-keys -t '{session_name}:{window['index']}' 'cd {win_path} && {win_cmd}' C-m",
                    check=False
                )

        print(f"[✓] 已恢复会话: '{session_name}' ({len(windows)} 个窗口)")
        restored += 1

    print(f"\n========================================")
    print(f"恢复完成: {restored} 个会话已恢复, {skipped} 个已存在")
    print(f"========================================")

    # 如果有活跃会话，提示如何 attach
    stdout, _ = run_cmd("tmux ls 2>/dev/null")
    if stdout:
        print("\n当前 tmux 会话:")
        print(stdout)
        print(f"\n连接命令: tmux attach -t <session_name>")

    return True


if __name__ == "__main__":
    save_file = sys.argv[1] if len(sys.argv) > 1 else None
    restore_workspace(save_file)
