#!/bin/bash
# ==============================================================================
#      AI 招聘助理 v4.1 - 启动器 (纯API + HF Embedding 核心)
# ==============================================================================
#
# v4.1 启动器特性:
# - [AI 引擎] 依赖本地运行的 OpenAI 兼容 API 服务 (如 Ollama, vLLM 等)。
# - [Embedding] 不再依赖任何外部服务，由 Python 脚本自动下载和管理。
# - [网络代理] 内置代理配置，确保 Notion API 能正常访问。
#
# ==============================================================================
 
# --- 1. 用户配置区 ---
PYTHON_SCRIPT_NAME="14.py"       # 您的主 Python 脚本文件名
# 【【【请确认】】】 您的本地AI服务是否正在运行，脚本会连接到它
LOCAL_API_SERVER_COMMAND="ollama serve"          # 这是一个示例命令，您可以留空

CENTRAL_DB_CONTAINER_NAME="ai_database_hub"      # Qdrant Docker 容器名
VENV_DIR="venv"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PROXY_URL="http://127.0.0.1:2080" # Notion API 需要代理

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    AI 招聘助理 v4.1 - 启动器 (纯API + HF Embedding)"
echo "========================================================"
echo ""

# --- 步骤 1: 检查核心系统依赖 ---
echo "[1/4] 正在检查核心系统依赖 (docker, python3)..."
dependencies=("docker" "python3")
for cmd in "${dependencies[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ 严重错误: 未找到核心命令 '$cmd'。请先安装它。"
        read -p "按任意键退出..."
        exit 1
    fi
done
echo "✅ 所有核心系统依赖已就绪。"
echo ""

# --- 步骤 2: 配置 Python 虚拟环境与依赖 ---
echo "[2/4] 正在配置 Python 虚拟环境并安装依赖..."
if [ ! -d "$VENV_DIR" ]; then
    echo ">> 虚拟环境不存在，正在创建..."
    python3 -m venv $VENV_DIR
fi
source "${VENV_DIR}/bin/activate"

if [ ! -f "requirements.txt" ]; then
    echo "❌ 严重错误: 未找到 'requirements.txt' 文件！"
    read -p "按任意键退出..."
    exit 1
fi
pip install --upgrade pip > /dev/null 2>&1
pip install -i "${PIP_MIRROR}" -r requirements.txt
echo "✅ Python 环境已就绪。"
echo ""

# --- 步骤 3: 检查后台服务 (数据库 & AI API) ---
echo "[3/4] 正在检查后台服务..."
# 检查Qdrant数据库
if ! docker ps -q -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ &> /dev/null; then
    if docker ps -aq -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ &> /dev/null; then
        echo ">> [Qdrant] 检测到已停止的数据库，正在重启..."
        docker start ${CENTRAL_DB_CONTAINER_NAME} > /dev/null 2>&1
    else
        echo "❌ 严重错误: 未找到Docker容器 '${CENTRAL_DB_CONTAINER_NAME}'！"
        read -p "按任意键退出..."
        exit 1
    fi
fi
echo "✅ [Qdrant] 数据库服务已就绪。"

# 检查本地 AI API 服务是否在运行
# 注意：这是一个简单的检查，可能不完全准确，但能提醒用户
if ! curl -s http://127.0.0.1:8087/v1/models > /dev/null; then
    echo "⚠️ 警告: 无法连接到 http://127.0.0.1:8087/v1。"
    echo "   请确保您的本地 AI API 服务 (如 Ollama, vLLM, LM Studio) 正在运行，"
    echo "   并且监听的是这个地址和端口，否则 Python 脚本会启动失败。"
fi
echo "✅ [AI API] 请确保您的本地 AI 服务正在后台运行。"
echo ""

# --- 步骤 4: 启动主程序 ---
echo "[4/4] 正在启动 Python 主程序 (${PYTHON_SCRIPT_NAME})..."
if [ -n "$PROXY_URL" ]; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export NO_PROXY="localhost,127.0.0.1"
    echo "✅ 网络代理已激活: ${PROXY_URL} (用于Notion等)"
    echo "✅ 本地服务直连已配置。"
else
    echo "⚠️ 警告: 未配置代理。如果 Notion API 无法访问，请在脚本中配置 PROXY_URL。"
fi

echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""
python3 "${PYTHON_SCRIPT_NAME}"
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
deactivate
read -p "脚本执行完毕，按回车键退出..."```

