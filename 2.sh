#!/bin/bash
# ==============================================================================
#      AI 简历信息提取助理 v1.0 - 启动器 (防闪退 & 防代理冲突版)
# ==============================================================================
# v1.2 更新:
# - 增加了 NO_PROXY 环境变量，以解决系统 SOCKS 代理与程序 HTTP 代理的冲突。
# - 确保连接本地服务 (Ollama, Qdrant) 时不走任何代理。
# ==============================================================================

# --- 1. 用户配置区 ---
PYTHON_SCRIPT_NAME="12.py" 
CENTRAL_DB_CONTAINER_NAME="ai_database_hub"
VENV_DIR="venv"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PROXY_URL="http://127.0.0.1:2080"

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    AI 简历信息提取助理 v1.0 - 启动器"
echo "========================================================"
echo ""

# --- 步骤 1: 检查核心系统依赖 ---
echo "[1/5] 正在检查核心系统依赖 (docker, python3, ollama)..."
dependencies=("docker" "python3" "ollama")
for cmd in "${dependencies[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ 错误: 未找到核心命令 '$cmd'。请先安装它。"
        read -p "按任意键退出..."
        exit 1
    fi
done
echo "✅ 所有核心系统依赖已安装。"
echo ""

# --- 步骤 2: 配置 Python 虚拟环境 ---
echo "[2/5] 正在配置 Python 虚拟环境 (${VENV_DIR})..."
if [ ! -d "$VENV_DIR" ]; then
    echo ">> 虚拟环境不存在，正在创建..."
    python3 -m venv $VENV_DIR
fi
source "${VENV_DIR}/bin/activate"
echo "✅ 虚拟环境已激活。"
echo ""

# --- 步骤 3: 安装项目依赖库 ---
echo "[3/5] 正在安装 Python 依赖库..."
pip install --upgrade pip > /dev/null 2>&1
pip install -i "${PIP_MIRROR}" -r requirements.txt
echo "✅ 所有 Python 依赖库已安装/更新。"
echo ""

# --- 步骤 4: 检查后台服务 ---
echo "[4/5] 正在检查后台服务 (数据库 & AI引擎)..."
if ! docker ps -q -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ &> /dev/null; then
    if docker ps -aq -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ &> /dev/null; then
        docker start ${CENTRAL_DB_CONTAINER_NAME} > /dev/null 2>&1
    fi
fi
echo "✅ [DB Hub] 数据库服务已就绪。"
if ! ollama ps &> /dev/null; then
    echo "❌ 错误: Ollama 服务未运行。请先启动 Ollama 服务！"
    read -p "按任意键退出..."
    exit 1
fi
models_to_check=("gemma3:4b" "embeddinggemma") 
for model in "${models_to_check[@]}"; do
    if ! ollama list | grep -q "${model}"; then
        echo "⚠️ 警告: 未找到核心模型 '${model}'。请运行 'ollama pull ${model}' 下载。"
        read -p "下载完成后，请按任意键继续..."
    fi
done
echo "✅ [Ollama] AI 引擎及模型已就绪。"
echo ""

# --- 步骤 5: 启动主程序 ---
echo "[5/5] 正在启动 Python 主程序..."
if [ -n "$PROXY_URL" ]; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    # 【【【 关键修复：增加这一行！！！ 】】】
    # 这会告诉所有程序，当访问 localhost 和 127.0.0.1 时，不要使用任何代理。
    export NO_PROXY="localhost,127.0.0.1"
    
    echo "✅ 代理已激活: ${PROXY_URL}"
    echo "✅ 本地服务直连已配置。"
else
    echo "✅ 未配置代理，将尝试直接连接。"
fi

echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""

python3 "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
deactivate
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
read -p "脚本执行完毕，按回车键退出..."