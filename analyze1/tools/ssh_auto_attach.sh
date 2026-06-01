#!/bin/bash
# SSH Auto Attach — VPN恢复后一键连接服务器并恢复工作区
# 用法: ./ssh_auto_attach.sh [session_name]

SERVER_IP="${SERVER_IP:-REDACTED_IP}"
SERVER_USER="${SERVER_USER:-test}"
TMUX_SESSION="${1:-csco}"

echo "========================================"
echo "自动连接服务器并恢复工作区"
echo "========================================"
echo "目标: ${SERVER_USER}@${SERVER_IP}"
echo "会话: ${TMUX_SESSION}"
echo ""

# 先保存本地工作区快照
SAVE_DIR="${HOME}/.vpn_workspace_saves"
mkdir -p "$SAVE_DIR"
echo "timestamp: $(date)" > "${SAVE_DIR}/last_attach.txt"

# SSH + 自动恢复 tmux 会话 / 检查管线状态
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -t "${SERVER_USER}@${SERVER_IP}" bash -s << EOF
    echo "=== 服务器工作区恢复 ==="
    echo "时间: \$(date)"
    echo ""

    # 检查 tmux 会话
    echo "--- tmux 会话 ---"
    tmux ls 2>/dev/null || echo "(无活跃会话)"
    echo ""

    # 检查管线进度
    if [ -f ~/CSC-O/work/pipeline_state.json ]; then
        echo "--- 管线进度 ---"
        python3 -c "
import json
with open(os.path.expanduser('~/CSC-O/work/pipeline_state.json')) as f:
    s = json.load(f)
print('已完成阶段:', [x['name'] for x in s.get('completed_stages', [])])
print('当前阶段:', s.get('current_stage', '无'))
print('错误数:', len(s.get('errors', [])))
" 2>/dev/null || cat ~/CSC-O/work/pipeline_state.json 2>/dev/null | head -20
        echo ""
    fi

    # 检查运行日志
    if [ -f ~/CSC-O/output/run.log ]; then
        echo "--- 最近日志 (10行) ---"
        tail -10 ~/CSC-O/output/run.log
        echo ""
    fi

    # 尝试 attach 到指定会话
    echo "--- 正在 attach 到会话: ${TMUX_SESSION} ---"
    tmux attach -t ${TMUX_SESSION} || tmux new -s ${TMUX_SESSION}
EOF
