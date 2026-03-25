#!/bin/bash

# =================================================================
#      AI 招聘助理 v35.0 - 启动器 (虚拟环境终极版)
# =================================================================
# v5.0 更新:
# - 【核心修复】集成了正确的虚拟环境(venv)创建、激活和依赖安装流程。
# - 结合了Docker检查和Python环境准备，成为一个完整的启动方案。
# - 确保程序在隔离且配置正确的环境中运行，彻底解决ModuleNotFoundError。
# =================================================================

# --- 配置区 ---
PYTHON_SCRIPT_NAME="17.py"
CENTRAL_DB_CONTAINER_NAME="ai_database_hub"

# --- 自动切换到脚本所在目录 ---
cd "$(dirname "$0")" || exit

# --- 主逻辑 ---
clear
echo "========================================================"
echo "      AI 招聘助理 v35.0 - 正在启动 (纯本地模式)..."
echo "========================================================"
echo ""

# --- 步骤 1: 检查中央数据库 ---
echo "[1/3] 正在检查中央数据库 (${CENTRAL_DB_CONTAINER_NAME}) 状态..."

if [ -n "$(docker ps -q -f name=^/${CENTRAL_DB_CONTAINER_NAME}$)" ]; then
    echo "      > 状态: ✅ 中央数据库正在运行。"
elif [ -n "$(docker ps -aq -f name=^/${CENTRAL_DB_CONTAINER_NAME}$)" ]; then
    echo "      > 状态: 检测到已停止的中央数据库，正在重启..."
    docker start ${CENTRAL_DB_CONTAINER_NAME}
    echo "      > 状态: ✅ 中央数据库已成功启动。"
else
    echo "      > 状态: ❌ 严重错误: 未找到中央数据库 '${CENTRAL_DB_CONTAINER_NAME}'！"
    echo "      >      请先运行创建数据库的 'docker run...' 命令。"
    echo ""
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 步骤 2: 准备 Python 虚拟环境 & 依赖 ---
echo "[2/3] 正在准备 Python 虚拟环境并安装依赖..."
# 检查名为 'venv' 的文件夹是否存在，如果不存在则创建
if [ ! -d "venv" ]; then
    echo "      > 虚拟环境不存在，正在自动创建..."
    python3 -m venv venv
fi
# 激活虚拟环境
source "venv/bin/activate"

# 检查 requirements.txt 文件是否存在
if [ ! -f "requirements.txt" ]; then
    echo "      > ❌ 严重错误: 未找到 'requirements.txt' 文件！"
    read -p "按任意键退出..."
    exit 1
fi
# 在虚拟环境中安装依赖，使用清华镜像源加速，并将输出重定向以保持界面整洁
pip install -i "https://pypi.tuna.tsinghua.edu.cn/simple" -r requirements.txt > /dev/null
echo "      > 状态: ✅ Python 环境已在虚拟环境中就绪。"
echo ""

# --- 步骤 3: 启动 Python 主程序 ---
echo "[3/3] 正在启动 Python 主程序 (${PYTHON_SCRIPT_NAME})..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""

# 运行Python脚本 (此时会使用venv中的python解释器)
python3 "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束后的提示 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
# 退出虚拟环境
deactivate
echo ""
echo "========================================================"
echo "               主程序已执行完毕或关闭"
echo "========================================================"
echo "[提示] 中央数据库 (${CENTRAL_DB_CONTAINER_NAME}) 仍在后台运行。"
echo ""

read -p "所有任务已完成，按任意键退出此启动器窗口..."