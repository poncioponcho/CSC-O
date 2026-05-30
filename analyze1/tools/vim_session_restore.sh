#!/bin/bash
# Vim Session Restore — 从保存的 vim 会话恢复编辑状态
# 用法: ./vim_session_restore.sh [session_name]

SESSION_NAME="${1:-csco}"
SAVE_DIR="${HOME}/.vpn_workspace_saves/vim_sessions"
SAVE_FILE="${SAVE_DIR}/${SESSION_NAME}_latest.viminfo"

echo "========================================"
echo "Vim 会话恢复: $SESSION_NAME"
echo "========================================"

if [ ! -f "$SAVE_FILE" ]; then
    echo "[!] 未找到保存文件: $SAVE_FILE"
    echo "    请先运行: ./vim_session_save.sh $SESSION_NAME"
    exit 1
fi

# 解析保存文件并重建 pane
echo "[*] 从 $SAVE_FILE 恢复..."

# 简单方式: 在当前会话中重新打开文件
# 由于 tmux pane 重建复杂，这里采用发送命令到现有 pane 的方式

# 检查会话是否存在
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[!] tmux 会话 '$SESSION_NAME' 不存在"
    echo "    先恢复工作区: python3 tools/workspace_restore.py"
    exit 1
fi

# 读取保存文件并发送恢复命令
pane_idx=0
while true; do
    ptype=$(grep "^pane_${pane_idx}_type:" "$SAVE_FILE" | cut -d: -f2 | xargs)
    [ -z "$ptype" ] && break

    pid=$(grep "^pane_${pane_idx}_id:" "$SAVE_FILE" | cut -d: -f2 | xargs)
    ppath=$(grep "^pane_${pane_idx}_path:" "$SAVE_FILE" | cut -d: -f2 | xargs)
    pfile=$(grep "^pane_${pane_idx}_file:" "$SAVE_FILE" | cut -d: -f2 | xargs)
    pcmd=$(grep "^pane_${pane_idx}_cmd:" "$SAVE_FILE" | cut -d: -f2 | xargs)

    if [ "$ptype" = "vim" ] && [ -n "$pfile" ] && [ "$pfile" != "unknown" ]; then
        # 发送 vim 恢复命令到对应 pane
        # 使用 tmux send-keys 发送 cd + vim 命令
        target_pane="${SESSION_NAME}.${pane_idx}"
        if tmux list-panes -t "$SESSION_NAME" | grep -q "^${pane_idx}:"; then
            tmux send-keys -t "$target_pane" "cd $ppath && vim '$pfile'" C-m
            echo "[✓] 恢复 vim: $pfile (pane $pane_idx)"
        fi
    elif [ "$ptype" = "shell" ] && [ -n "$ppath" ]; then
        target_pane="${SESSION_NAME}.${pane_idx}"
        if tmux list-panes -t "$SESSION_NAME" | grep -q "^${pane_idx}:"; then
            tmux send-keys -t "$target_pane" "cd $ppath" C-m
            echo "[~] 恢复目录: $ppath (pane $pane_idx)"
        fi
    fi

    pane_idx=$((pane_idx + 1))
done

echo ""
echo "========================================"
echo "Vim 会话恢复完成"
echo "========================================"
echo "提示: 如果 vim 文件未正确恢复，请检查文件是否仍存在。"
