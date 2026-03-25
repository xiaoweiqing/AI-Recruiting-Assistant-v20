#!/bin/bash
# ==============================================================================
#      AI 简历信息提取助理 v1.1 - 启动器 (Llama.cpp 核心版)
# ==============================================================================
# v1.1 更新:
# - 核心依赖检查从 Ollama 切换为 Llama.cpp 可执行文件。
# - Ollama 服务检查被降级为可选的“向量化服务”，且只检查 embedding 模型。
# ==============================================================================
 
# --- 1. 用户配置区 ---
PYTHON_SCRIPT_NAME="13.py" # <-- 建议使用更有意义的文件名
LLAMACPP_EXECUTABLE="/home/weiyubin/llama.cpp/build/bin/llama-cli" # 【【【 新增：Llama.cpp 程序路径 】】】

CENTRAL_DB_CONTAINER_NAME="ai_database_hub"
VENV_DIR="venv"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PROXY_URL="http://127.0.0.1:2080"

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    AI 简历信息提取助理 v1.1 - 启动器 (Llama.cpp 核心)"
echo "========================================================"
echo ""

# --- 步骤 1: 检查核心系统依赖 ---
echo "[1/5] 正在检查核心系统依赖 (docker, python3, llama.cpp)..."

# 检查基础命令
dependencies=("docker" "python3")
for cmd in "${dependencies[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ 错误: 未找到核心命令 '$cmd'。请先安装它。"
        read -p "按任意键退出..."
        exit 1
    fi
done

# 【【【 修改：检查 Llama.cpp 文件是否存在 】】】
if [ ! -f "$LLAMACPP_EXECUTABLE" ]; then
    echo "❌ 错误: 未在以下路径找到 Llama.cpp 程序: "
    echo "   -> ${LLAMACPP_EXECUTABLE}"
    echo "   -> 请检查脚本中的 LLAMACPP_EXECUTABLE 路径配置是否正确。"
    read -p "按任意键退出..."
    exit 1
fi

echo "✅ 所有核心系统依赖已就绪。"
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
# 确保 requirements.txt 里有需要的库
pip install -i "${PIP_MIRROR}" -r requirements.txt
echo "✅ 所有 Python 依赖库已安装/更新。"
echo ""

# --- 步骤 4: 检查后台服务 ---
echo "[4/5] 正在检查后台服务 (数据库 & 向量化服务)..."
# 检查数据库
if ! docker ps -q -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ &> /dev/null; then
    if docker ps -aq -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ &> /dev/null; then
        docker start ${CENTRAL_DB_CONTAINER_NAME} > /dev/null 2>&1
    fi
fi
echo "✅ [DB Hub] 数据库服务已就绪。"

# 【【【 修改：Ollama 现在是可选的向量化服务 】】】
echo ">> 正在检查可选的 Ollama 向量化服务..."
if ! ollama ps &> /dev/null; then
    echo "⚠️ 警告: Ollama 服务未运行。简历向量化和RAG搜索功能将不可用。"
    echo "   (如果不需要此功能，可忽略此警告)"
else
    # 只检查 embedding 模型
    if ! ollama list | grep -q "embeddinggemma"; then
        echo "⚠️ 警告: 未找到向量化模型 'embeddinggemma'。请运行 'ollama pull embeddinggemma' 下载。"
        echo "   (否则向量化功能将失败)"
    fi
    echo "✅ [Ollama] 向量化服务已就绪。"
fi
echo ""

# --- 步骤 5: 启动主程序 ---
echo "[5/5] 正在启动 Python 主程序 (${PYTHON_SCRIPT_NAME})..."
if [ -n "$PROXY_URL" ]; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export NO_PROXY="localhost,127.0.0.1" # 确保本地服务直连
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