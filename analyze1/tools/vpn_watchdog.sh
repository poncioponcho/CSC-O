#!/bin/bash
# VPN Watchdog Pro — VPN断连监测、预测与自动恢复系统 (Mac端)
# 功能:
#   1. 实时监测 VPN 连接状态
#   2. 基于延迟趋势预测即将断连 (延迟陡增/丢包率上升时提前预警)
#   3. VPN 断开时触发本地+远程工作区保存
#   4. VPN 恢复后自动 SSH + 恢复 tmux 会话
#
# 用法:
#   前台: ./vpn_watchdog.sh
#   后台: nohup ./vpn_watchdog.sh > ~/.vpn_watchdog.log 2>&1 &
#   停止: kill $(cat ~/.vpn_watchdog.pid)

# ═══════════════════════════════════════════════════════════════
# 配置区
# ═══════════════════════════════════════════════════════════════
SERVER_IP="${VPNWD_SERVER_IP:-49.52.29.0}"
SERVER_USER="${VPNWD_SERVER_USER:-test}"
TMUX_SESSION="${VPNWD_SESSION:-csco}"
CHECK_INTERVAL="${VPNWD_INTERVAL:-5}"
LOG_FILE="${HOME}/.vpn_watchdog.log"
PID_FILE="${HOME}/.vpn_watchdog.pid"
LATENCY_HISTORY_FILE="${HOME}/.vpn_watchdog_latency"

# 预测阈值
LATENCY_SPIKE_THRESHOLD=300     # 延迟陡增阈值 (ms, 超过平均值 3 倍)
PACKET_LOSS_WARN=20             # 丢包率预警阈值 (%)
CONSECUTIVE_FAILS=3             # 连续失败次数判定为断开

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

notify_mac() {
    local title="$1"
    local msg="$2"
    if command -v osascript &> /dev/null; then
        osascript -e "display notification \"$msg\" with title \"$title\"" 2>/dev/null
    fi
}

detect_vpn_interface() {
    ifconfig | awk '/^utun[0-9]+:/{iface=$1} /inet [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/ && iface != "" {gsub(/:/,"",iface); print iface; exit}'
}

get_vpn_gateway() {
    local iface="$1"
    netstat -rn | grep "^default" | grep "$iface" | awk '{print $2}' | head -1
}

# 测量到 VPN 网关的延迟，同时记录历史用于趋势预测
measure_latency() {
    local target="$1"
    local latency=""
    local loss=0

    if [ -n "$target" ]; then
        # 发送 3 个 ping，解析结果
        local ping_out
        ping_out=$(ping -c 3 -W 2 "$target" 2>/dev/null)
        # 提取平均延迟
        latency=$(echo "$ping_out" | tail -1 | grep -o 'avg [0-9.]*' | awk '{print $2}')
        # 提取丢包率
        local transmitted received
        transmitted=$(echo "$ping_out" | grep -o '[0-9]* packets transmitted' | awk '{print $1}')
        received=$(echo "$ping_out" | grep -o '[0-9]* received' | awk '{print $1}')
        if [ -n "$transmitted" ] && [ "$transmitted" -gt 0 ]; then
            loss=$(( (transmitted - received) * 100 / transmitted ))
        fi
    fi

    # 如果 ping 失败，尝试测量到服务器的延迟作为备用
    if [ -z "$latency" ]; then
        local ping_out2
        ping_out2=$(ping -c 1 -W 2 "$SERVER_IP" 2>/dev/null)
        latency=$(echo "$ping_out2" | grep 'time=' | head -1 | sed -n 's/.*time=\([0-9.]*\).*/\1/p')
    fi

    echo "${latency:-9999} ${loss:-100}"
}

# 基于历史延迟数据预测连接稳定性
analyze_trend() {
    local history_file="$1"
    if [ ! -f "$history_file" ] || [ "$(wc -l < "$history_file")" -lt 5 ]; then
        echo "insufficient_data"
        return
    fi

    # 计算最近 10 个样本的平均值和标准差
    local avg stddev max_val
    avg=$(awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count}' "$history_file")
    stddev=$(awk -v m="$avg" '{sum+=($1-m)^2} END {if(NR>0) printf "%.1f", sqrt(sum/NR)}' "$history_file")
    max_val=$(awk 'BEGIN{max=0} {if($1>max) max=$1} END{print max}' "$history_file")

    # 最新值
    local latest
    latest=$(tail -1 "$history_file" | awk '{print $1}')

    # 判定逻辑
    if [ "${latest%.*}" -gt 5000 ]; then
        echo "disconnected"
    elif awk "BEGIN {exit !($latest > $avg * 3 && $latest > $LATENCY_SPIKE_THRESHOLD)}"; then
        echo "spike"
    elif awk "BEGIN {exit !($stddev > $avg * 0.8)}"; then
        echo "unstable"
    else
        echo "stable"
    fi
}

# 远程触发服务器端工作区保存
remote_save_workspace() {
    log "[SAVE] 触发远程工作区保存..."
    ssh -o ConnectTimeout=5 "${SERVER_USER}@${SERVER_IP}" \
        "cd ~/CSC-O && source ~/csco_env/bin/activate 2>/dev/null; python3 tools/workspace_save.py 2>/dev/null || echo 'workspace_save.py not found'" \
        > /dev/null 2>&1 &
}

auto_ssh_attach() {
    log "[RECOVER] 正在自动连接服务器并恢复 tmux 会话..."
    notify_mac "VPN Watchdog" "VPN 已恢复，正在自动连接服务器..."

    if command -v osascript &> /dev/null; then
        osascript <<EOF
            tell application "Terminal"
                if not (exists window 1) then reopen
                do script "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -t ${SERVER_USER}@${SERVER_IP} 'echo \"=== 恢复工作区 ===\"; python3 ~/CSC-O/tools/workspace_restore.py 2>/dev/null; tmux attach -t ${TMUX_SESSION} || tmux new -s ${TMUX_SESSION}'; exec bash"
                activate
            end tell
EOF
    else
        log "[RECOVER] 请手动执行: ssh -t ${SERVER_USER}@${SERVER_IP} 'tmux attach -t ${TMUX_SESSION} || tmux new -s ${TMUX_SESSION}'"
    fi
}

save_local_workspace() {
    local save_dir="${HOME}/.vpn_workspace_saves"
    mkdir -p "$save_dir"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local save_file="${save_dir}/mac_save_${timestamp}.txt"
    {
        echo "# Mac 本地工作区快照"
        echo "timestamp: $(date)"
        echo ""
        echo "# 当前目录"
        pwd
        echo ""
        echo "# 最近命令历史"
        history | tail -20 2>/dev/null || true
    } > "$save_file"
    log "[SAVE] Mac 本地工作区已快照: $save_file"
}

# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

main() {
    if [ -f "$PID_FILE" ]; then
        local old_pid
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "[!] vpn_watchdog 已在运行 (PID: $old_pid)"
            echo "    日志: $LOG_FILE"
            echo "    停止: kill $old_pid"
            exit 1
        fi
    fi
    echo $$ > "$PID_FILE"

    # 清理历史文件
    > "$LATENCY_HISTORY_FILE"

    log "========================================"
    log "VPN Watchdog Pro 启动"
    log "SERVER=${SERVER_USER}@${SERVER_IP}, SESSION=${TMUX_SESSION}, INTERVAL=${CHECK_INTERVAL}s"
    log "预测阈值: 延迟陡增>${LATENCY_SPIKE_THRESHOLD}ms, 丢包预警>${PACKET_LOSS_WARN}%"
    log "========================================"
    notify_mac "VPN Watchdog" "监测守护进程已启动"

    local vpn_was_down=0
    local disconnect_time=""
    local fail_count=0
    local warned_unstable=0

    while true; do
        local iface
        iface=$(detect_vpn_interface)

        if [ -n "$iface" ]; then
            local gateway
            gateway=$(get_vpn_gateway "$iface")
            local lat_loss
            lat_loss=$(measure_latency "$gateway")
            local latency=$(echo "$lat_loss" | awk '{print $1}')
            local loss=$(echo "$lat_loss" | awk '{print $2}')

            # 记录历史 (保留最近 20 条)
            echo "$latency $loss" >> "$LATENCY_HISTORY_FILE"
            tail -n 20 "$LATENCY_HISTORY_FILE" > "${LATENCY_HISTORY_FILE}.tmp"
            mv "${LATENCY_HISTORY_FILE}.tmp" "$LATENCY_HISTORY_FILE"

            local trend
            trend=$(analyze_trend "$LATENCY_HISTORY_FILE")

            # 预测性预警
            if [ "$trend" = "spike" ] && [ "$warned_unstable" -eq 0 ]; then
                log "[WARN] 检测到延迟陡增 (latest=${latency}ms)，即将断连风险高！"
                notify_mac "VPN Watchdog" "延迟陡增！即将触发保存..."
                save_local_workspace
                remote_save_workspace
                warned_unstable=1
            elif [ "$trend" = "unstable" ] && [ "$warned_unstable" -eq 0 ]; then
                log "[WARN] 网络不稳定，延迟波动大"
                notify_mac "VPN Watchdog" "网络不稳定"
                warned_unstable=1
            elif [ "$loss" -ge "$PACKET_LOSS_WARN" ] && [ "$warned_unstable" -eq 0 ]; then
                log "[WARN] 丢包率 ${loss}%，即将断连风险！"
                notify_mac "VPN Watchdog" "高丢包率！正在保存..."
                save_local_workspace
                remote_save_workspace
                warned_unstable=1
            fi

            # 连续失败计数器归零
            if [ "${latency%.*}" -lt 5000 ]; then
                fail_count=0
            fi

            # VPN 恢复处理
            if [ "$vpn_was_down" -eq 1 ]; then
                local reconnect_time
                reconnect_time=$(date '+%Y-%m-%d %H:%M:%S')
                log "[RECONNECT] VPN 已恢复 (断开: $disconnect_time, 恢复: $reconnect_time)"
                notify_mac "VPN Watchdog" "VPN 已恢复"
                sleep 3
                auto_ssh_attach
                vpn_was_down=0
                warned_unstable=0
                > "$LATENCY_HISTORY_FILE"
            fi
        else
            # 无 VPN 接口
            fail_count=$((fail_count + 1))
            if [ "$fail_count" -ge "$CONSECUTIVE_FAILS" ] && [ "$vpn_was_down" -eq 0 ]; then
                disconnect_time=$(date '+%Y-%m-%d %H:%M:%S')
                log "[DISCONNECT] VPN 断开于 $disconnect_time (连续 ${fail_count} 次检测失败)"
                log "[INFO] 服务器上的 tmux 会话仍继续运行，pipeline 进度由 --resume 保障"
                notify_mac "VPN Watchdog" "VPN 已断开，服务器程序仍在运行"
                save_local_workspace
                remote_save_workspace
                vpn_was_down=1
                warned_unstable=0
            fi
        fi

        sleep "$CHECK_INTERVAL"
    done
}

cleanup() {
    rm -f "$PID_FILE"
    log "[EXIT] VPN Watchdog 已停止"
    exit 0
}
trap cleanup EXIT INT TERM

main "$@"
