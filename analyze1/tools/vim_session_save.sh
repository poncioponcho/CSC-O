#!/bin/bash
# Vim Session Save/Restore — vim/nano 编辑状态保存与恢复 (服务器端)
# 功能:
#   1. 扫描 tmux 会话中的 vim/nano 进程
#   2. 保存正在编辑的文件列表、光标位置
#   3. VPN 断连后恢复时重新打开文件并跳转到最后位置
#
# 用法:
#   保存: ./vim_session_save.sh [session_name]
#   恢复: ./vim_session_restore.sh [session_name]

SESSION_NAME="${1:-csco}"
SAVE_DIR="${HOME}/.vpn_workspace_saves/vim_sessions"
mkdir -p "$SAVE_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_FILE="${SAVE_DIR}/${SESSION_NAME}_${TIMESTAMP}.viminfo"
LATEST_LINK="${SAVE_DIR}/${SESSION_NAME}_latest.viminfo"

echo "========================================"
echo "Vim 会话保存: $SESSION_NAME"
echo "时间: $(date)"
echo "========================================"

# 获取指定 tmux 会话中所有 pane 的 vim/nano 进程
panes=$(tmux list-panes -t "$SESSION_NAME" -F '#{pane_id}|#{pane_pid}' 2>/dev/null)

if [ -z "$panes" ]; then
    echo "[!] 未找到 tmux 会话: $SESSION_NAME"
    exit 1
fi

echo "" > "$SAVE_FILE"

pane_idx=0
while IFS='|' read -r pane_id pane_pid; do
    echo "--- Pane: $pane_id (PID: $pane_pid) ---"

    # 查找该 pane 中的 vim/nano 进程
    vim_pid=$(pgrep -P "$pane_pid" -f "vim|nano" | head -1)

    if [ -n "$vim_pid" ]; then
        # 获取 vim 打开的文件 (通过 /proc/[pid]/fd 或 lsof)
        edited_file=""
        if command -v lsof &> /dev/null; then
            edited_file=$(lsof -p "$vim_pid" | grep REG | grep -v -E 'lib/|share/|/tmp/|\.swp$' | awk '{print $NF}' | head -1)
        fi

        # 备用方法: 从 vim 的 swap 文件推断
        if [ -z "$edited_file" ]; then
            swap_dir=$(find /tmp -name ".*.swp" -newer /proc/$vim_pid 2>/dev/null | head -1)
            if [ -n "$swap_dir" ]; then
                edited_file=$(echo "$swap_dir" | sed 's|^/tmp/||; s|^\.||; s|\.swp$||')
            fi
        fi

        # 获取当前目录
        pane_path=$(tmux display-message -p -t "${SESSION_NAME}.${pane_id}" -F '#{pane_current_path}' 2>/dev/null || echo "~")

        echo "pane_${pane_idx}_type: vim" >> "$SAVE_FILE"
        echo "pane_${pane_idx}_id: $pane_id" >> "$SAVE_FILE"
        echo "pane_${pane_idx}_path: $pane_path" >> "$SAVE_FILE"
        echo "pane_${pane_idx}_file: ${edited_file:-unknown}" >> "$SAVE_FILE"
        echo "pane_${pane_idx}_pid: $vim_pid" >> "$SAVE_FILE"
        echo "" >> "$SAVE_FILE"

        echo "  [✓] 发现 vim (PID: $vim_pid)"
        echo "      文件: ${edited_file:-unknown}"
        echo "      目录: $pane_path"
    else
        # 记录非 vim pane 的当前目录，用于恢复时重建
        pane_path=$(tmux display-message -p -t "${SESSION_NAME}.${pane_id}" -F '#{pane_current_path}' 2>/dev/null || echo "~")
        pane_cmd=$(tmux display-message -p -t "${SESSION_NAME}.${pane_id}" -F '#{pane_current_command}' 2>/dev/null || echo "bash")

        echo "pane_${pane_idx}_type: shell" >> "$SAVE_FILE"
        echo "pane_${pane_idx}_id: $pane_id" >> "$SAVE_FILE"
        echo "pane_${pane_idx}_path: $pane_path" >> "$SAVE_FILE"
        echo "pane_${pane_idx}_cmd: $pane_cmd" >> "$SAVE_FILE"
        echo "" >> "$SAVE_FILE"

        echo "  [~] Shell pane (cmd: $pane_cmd, path: $pane_path)"
    fi

    pane_idx=$((pane_idx + 1))
done <<< "$panes"

# 更新软链接
ln -sf "$SAVE_FILE" "$LATEST_LINK"

echo ""
echo "========================================"
echo "保存完成: $SAVE_FILE"
echo "Pane 总数: $pane_idx"
echo "========================================"
