#!/bin/bash
# VPN 断连恢复系统 — 一键安装脚本
# 用法: ./install.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "VPN 断连自动保存与恢复系统 — 安装"
echo "========================================"

# 检测运行环境 (Mac 或 Linux)
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    PLATFORM="mac"
    echo "[*] 检测到 macOS"
elif [ "$OS" = "Linux" ]; then
    PLATFORM="linux"
    echo "[*] 检测到 Linux (服务器端)"
else
    echo -e "${RED}[!] 不支持的操作系统: $OS${NC}"
    exit 1
fi

# 赋予执行权限
echo "[*] 赋予脚本执行权限..."
chmod +x "$SCRIPT_DIR"/*.sh
chmod +x "$SCRIPT_DIR"/*.py 2>/dev/null || true

if [ "$PLATFORM" = "mac" ]; then
    echo "[*] Mac 端安装..."

    # 创建保存目录
    mkdir -p "$HOME/.vpn_workspace_saves"

    # 检查依赖
    if ! command -v osascript &> /dev/null; then
        echo -e "${YELLOW}[!] 警告: 未找到 osascript，自动恢复窗口功能不可用${NC}"
    fi

    echo ""
    echo -e "${GREEN}[✓] Mac 端安装完成${NC}"
    echo ""
    echo "启动命令:"
    echo "  cd $SCRIPT_DIR"
    echo "  ./vpn_watchdog.sh"
    echo ""
    echo "后台启动:"
    echo "  nohup ./vpn_watchdog.sh > ~/.vpn_watchdog.log 2>&1 &"
    echo ""
    echo "查看日志:"
    echo "  tail -f ~/.vpn_watchdog.log"

elif [ "$PLATFORM" = "linux" ]; then
    echo "[*] 服务器端安装..."

    # 检查 tmux
    if ! command -v tmux &> /dev/null; then
        echo -e "${YELLOW}[!] 警告: 未找到 tmux，请先安装: apt-get install tmux${NC}"
    fi

    # 检查 python3
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[!] 错误: 未找到 python3${NC}"
        exit 1
    fi

    # 创建保存目录
    mkdir -p "$HOME/.vpn_workspace_saves/vim_sessions"

    echo ""
    echo -e "${GREEN}[✓] 服务器端安装完成${NC}"
    echo ""
    echo "测试工作区保存:"
    echo "  cd $SCRIPT_DIR && python3 workspace_save.py"
    echo ""
    echo "测试工作区恢复:"
    echo "  cd $SCRIPT_DIR && python3 workspace_restore.py"
    echo ""
    echo "在 tmux 中启动 pipeline（推荐）:"
    echo "  tmux new -s csco"
    echo "  python $PROJECT_DIR/csco_pipeline.py --input data/xxx.csv --output output --work work --device cuda --resume"
fi

echo ""
echo "========================================"
echo "安装完成"
echo "========================================"
