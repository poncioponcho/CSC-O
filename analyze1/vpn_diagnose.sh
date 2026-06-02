#!/bin/bash
# VPN MTU 诊断与修复脚本
# 用法: ./vpn_diagnose.sh <目标服务器IP> [VPN接口名, 默认自动检测]
#
# 示例:
#   ./vpn_diagnose.sh REDACTED_IP
#   ./vpn_diagnose.sh REDACTED_IP utun4

set -e

TARGET_IP="${1:-}"
VPN_IF="${2:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "VPN MTU 诊断工具"
echo "========================================"
echo ""

# 检测 VPN 接口
if [ -z "$VPN_IF" ]; then
    echo "[*] 自动检测 VPN 接口..."
    VPN_IF=$(ifconfig | awk '/^utun[0-9]+:/{iface=$1} /inet 10\./ && iface != "" {gsub(/:/,"",iface); print iface; exit}')
    if [ -z "$VPN_IF" ]; then
        VPN_IF=$(ifconfig | awk '/^utun[0-9]+:/{iface=$1} /inet 172\./ && iface != "" {gsub(/:/,"",iface); print iface; exit}')
    fi
    if [ -z "$VPN_IF" ]; then
        VPN_IF=$(ifconfig | awk '/^utun[0-9]+:/{iface=$1} /inet 192\.168\./ && iface != "" {gsub(/:/,"",iface); print iface; exit}')
    fi
    if [ -z "$VPN_IF" ]; then
        echo -e "${RED}[!] 未检测到 VPN 接口 (utunX)。请先连接 VPN。${NC}"
        echo "    当前活跃接口:"
        ifconfig | grep -E "^utun[0-9]+:" | sed 's/^/    /'
        exit 1
    fi
fi

echo -e "${GREEN}[+] VPN 接口: $VPN_IF${NC}"

# 获取当前 MTU
CURRENT_MTU=$(ifconfig "$VPN_IF" | awk '/mtu /{print $4}')
echo "[*] 当前 $VPN_IF MTU: $CURRENT_MTU"

# 检查路由
VPN_ROUTE=$(netstat -rn | grep "^default" | grep "$VPN_IF" | head -1)
if [ -n "$VPN_ROUTE" ]; then
    echo -e "${GREEN}[+] 默认路由走 $VPN_IF${NC}"
else
    echo -e "${YELLOW}[!] 警告: 默认路由似乎不走 $VPN_IF${NC}"
    netstat -rn | grep "^default" | head -3
fi

# 如果没有提供目标 IP，提示输入
if [ -z "$TARGET_IP" ]; then
    echo ""
    echo "[*] 未提供目标服务器 IP。"
    echo "    如果你有 4090 服务器 IP，请重新运行:"
    echo "    ./vpn_diagnose.sh <服务器IP>"
    echo ""
    read -p "是否现在输入目标 IP 进行 MTU 测试? (或直接回车跳过): " TARGET_IP
    if [ -z "$TARGET_IP" ]; then
        echo "[*] 跳过 MTU 测试。通用修复命令:"
        echo "    sudo ifconfig $VPN_IF mtu 1280"
        exit 0
    fi
fi

echo ""
echo "[*] 测试到 $TARGET_IP 的 MTU 瓶颈..."
echo "    (使用 ping -D -s <size>，找到不丢包的最大值)"
echo ""

# MTU 测试
MAX_OK=0
for size in 1472 1400 1300 1280 1200 1100 1000; do
    MTU=$((size + 28))
    if ping -D -s "$size" -c 1 -W 2 "$TARGET_IP" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ MTU $MTU (size=$size) 通过${NC}"
        MAX_OK=$size
        break
    else
        echo -e "  ${RED}✗ MTU $MTU (size=$size) 失败${NC}"
    fi
done

if [ "$MAX_OK" -eq 0 ]; then
    echo -e "${RED}[!] 即使 MTU 1028 也 ping 不通。可能目标 IP 不可达或 ICMP 被过滤。${NC}"
    echo "    请确认目标 IP 正确且 VPN 已连接。"
    exit 1
fi

# 向上二分查找精确最大值
for size in 1380 1420 1440 1460; do
    if [ "$size" -lt "$MAX_OK" ]; then
        continue
    fi
    MTU=$((size + 28))
    if ping -D -s "$size" -c 1 -W 2 "$TARGET_IP" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ MTU $MTU (size=$size) 通过${NC}"
        MAX_OK=$size
    else
        echo -e "  ${RED}✗ MTU $MTU (size=$size) 失败${NC}"
        break
    fi
done

RECOMMENDED_MTU=$((MAX_OK + 28))
echo ""
echo -e "${GREEN}[+] 找到安全 MTU: $RECOMMENDED_MTU (payload size=$MAX_OK)${NC}"

# 提供修复选项
if [ "$CURRENT_MTU" -le "$RECOMMENDED_MTU" ]; then
    echo -e "${GREEN}[+] 当前 MTU ($CURRENT_MTU) 已经 <= 推荐值 ($RECOMMENDED_MTU)。${NC}"
    echo "    如果仍有问题，可能是其他原因（如 VPN 服务器端限速、防火墙等）。"
else
    echo ""
    echo -e "${YELLOW}[!] 当前 MTU ($CURRENT_MTU) > 推荐值 ($RECOMMENDED_MTU)。${NC}"
    echo "    这很可能导致 SSH banner / TLS 握手超时。"
    echo ""
    echo "    修复方式 (选一种):"
    echo ""
    echo "    [A] 立即临时修复 (当前会话有效，重启后失效):"
    echo "        sudo ifconfig $VPN_IF mtu $RECOMMENDED_MTU"
    echo ""
    echo "    [B] 重新连接 openconnect 时指定 MTU (推荐):"
    echo "        sudo openconnect --mtu $RECOMMENDED_MTU <你的VPN地址>"
    echo ""
    echo "    [C] 如果是 Cisco AnyConnect / openconnect 自动脚本，"
    echo "        在脚本中加入 mtu 参数。"
    echo ""
    read -p "是否立即执行临时修复 [A]? (y/n): " DO_FIX
    if [ "$DO_FIX" = "y" ] || [ "$DO_FIX" = "Y" ]; then
        echo "[*] 执行: sudo ifconfig $VPN_IF mtu $RECOMMENDED_MTU"
        sudo ifconfig "$VPN_IF" mtu "$RECOMMENDED_MTU"
        NEW_MTU=$(ifconfig "$VPN_IF" | awk '/mtu /{print $4}')
        echo -e "${GREEN}[+] $VPN_IF MTU 已调整为 $NEW_MTU${NC}"
        echo ""
        echo "[*] 重新测试 SSH 连接:"
        echo "    ssh -o ConnectTimeout=10 <USER>@$TARGET_IP"
        echo ""
        echo "[*] 如果仍然失败，尝试进一步降低 MTU:"
        echo "    sudo ifconfig $VPN_IF mtu 1280"
    fi
fi

echo ""
echo "========================================"
echo "诊断完成"
echo "========================================"
