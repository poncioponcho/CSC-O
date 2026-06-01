#!/bin/bash
# Check Server Status — Mac 端一键查询服务器 pipeline 运行状态
# 用法: ./check_server_status.sh

SERVER_IP="${SERVER_IP:-REDACTED_IP}"
SERVER_USER="${SERVER_USER:-test}"

echo "========================================"
echo "服务器运行状态查询"
echo "目标: ${SERVER_USER}@${SERVER_IP}"
echo "查询时间: $(date)"
echo "========================================"

ssh -o ConnectTimeout=10 "${SERVER_USER}@${SERVER_IP}" bash -s << 'EOF'
    echo ""
    echo "--- 1. 系统负载 ---"
    uptime

    echo ""
    echo "--- 2. GPU 状态 ---"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi 不可用"

    echo ""
    echo "--- 3. tmux 会话 ---"
    tmux ls 2>/dev/null || echo "(无 tmux 会话)"

    echo ""
    echo "--- 4. 管线进度 (pipeline_state.json) ---"
    if [ -f ~/CSC-O/work/pipeline_state.json ]; then
        python3 -c "
import json
with open(os.path.expanduser('~/CSC-O/work/pipeline_state.json')) as f:
    s = json.load(f)
completed = [x['name'] for x in s.get('completed_stages', [])]
current = s.get('current_stage', '无')
errors = s.get('errors', [])
print('已完成阶段:', completed if completed else '(无)')
print('当前阶段:', current)
print('错误数:', len(errors))
if errors:
    print('最近错误:', errors[-1].get('error', 'unknown')[:100])
" 2>/dev/null || cat ~/CSC-O/work/pipeline_state.json
    else
        echo "(无状态文件)"
    fi

    echo ""
    echo "--- 5. 运行日志 (最后15行) ---"
    if [ -f ~/CSC-O/output/run.log ]; then
        tail -15 ~/CSC-O/output/run.log
    else
        echo "(无日志文件)"
    fi

    echo ""
    echo "--- 6. 磁盘空间 ---"
    df -h /home /data 2>/dev/null | grep -E 'Filesystem|/home|/data'

    echo ""
    echo "--- 7. Python 进程 ---"
    ps aux | grep -E 'python.*csco' | grep -v grep || echo "(无 csco 相关 Python 进程)"
EOF

echo ""
echo "========================================"
echo "查询完成"
echo "========================================"
