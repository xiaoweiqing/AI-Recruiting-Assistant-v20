#!/bin/bash
# ==============================================================================
#      AI 招聘助理 v20.0 - 启动器 v3.1 (Ollama 最终版)
# ==============================================================================
# v3.1 更新:
# - 【代理修正】: 强制为 Python 程序设置正确的 HTTP 代理，覆盖系统环境。
# - 【Ollama集成】: 彻底移除 HuggingFace 模型下载逻辑。
# - 【环境检查】: 新增对 Ollama 服务和所需模型的自动检查。
# ==============================================================================

# --- 1. 用户配置区 ---
PYTHON_SCRIPT_NAME="4.py"
CENTRAL_DB_CONTAINER_NAME="ai_database_hub"
VENV_DIR="venv"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# 【【【 非常关键的一步：在这里填入您正确的HTTP代理地址 】】】
# !!! 请将 7890 替换为您代理软件的真实 HTTP 端口号 (例如 Clash 通常是 7890) !!!
# !!! 如果您不需要代理来访问 Notion，请将这一行留空，像这样: PROXY_URL=""
PROXY_URL="http://127.0.0.1:2080"

# --- 脚本主逻辑 ---
clear
echo "========================================================"
echo "    AI 招聘助理 v20.0 启动器 v3.1 (Ollama 最终版)"
echo "========================================================"
echo ""

# --- 步骤 1: 检查核心系统依赖 ---
echo "[1/6] 正在检查核心系统依赖..."
dependencies=("docker" "python3" "ollama")
for cmd in "${dependencies[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ 错误: 未找到核心命令 '$cmd'。请先安装它再运行此脚本。"
        read -p "按任意键退出..."
        exit 1
    fi
done
echo "✅ 所有核心系统依赖已安装。"
echo ""

# --- 步骤 2: 创建并激活 Python 虚拟环境 ---
echo "[2/6] 正在配置 Python 虚拟环境 (${VENV_DIR})..."
if [ ! -d "$VENV_DIR" ]; then
    echo ">> 虚拟环境不存在，正在为您创建..."
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 创建虚拟环境失败。"
        read -p "按任意键退出..."
        exit 1
    fi
fi
source "${VENV_DIR}/bin/activate"
echo "✅ 虚拟环境已激活。"
echo ""

# --- 步骤 3: 安装项目依赖库 ---
echo "[3/6] 正在安装 Python 依赖库 (使用高速镜像)..."
pip install -i "${PIP_MIRROR}" -r requirements.txt
echo "✅ 所有 Python 依赖库已安装/更新。"
echo ""

# --- 步骤 4: 检查并确保【中央数据库】正在运行 ---
echo "[4/6] 正在检查中央数据库 (${CENTRAL_DB_CONTAINER_NAME}) 状态..."
if ! docker ps -q -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ &> /dev/null; then
    if docker ps -aq -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ &> /dev/null; then
        echo ">> [DB Hub] 检测到已停止的数据库，正在重启..."
        docker start ${CENTRAL_DB_CONTAINER_NAME}
    else
        echo "❌ 严重错误: 未找到数据库容器 '${CENTRAL_DB_CONTAINER_NAME}'！"
        read -p "按任意键退出..."
        exit 1
    fi
fi
echo "✅ [DB Hub] 中央数据库正在运行。"
echo ""

# --- 步骤 5: 检查 Ollama 服务和所需模型 ---
echo "[5/6] 正在检查 Ollama 服务及所需模型..."
if ! ollama ps &> /dev/null; then
    echo "❌ 错误: Ollama 服务未运行。请先启动 Ollama 服务！"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ Ollama 服务正在运行。"
# 【【【 修改这里 】】】
models_to_check=("gemma3:4b" "embeddinggemma") 
for model in "${models_to_check[@]}"; do
    if ! ollama list | grep -q "${model}"; then
        echo "⚠️ 警告: 未找到模型 '${model}'。请在另一个终端运行 'ollama pull ${model}' 下载。"
        read -p "下载完成后，请按任意键继续..."
    fi
done
echo "✅ 所需的 Ollama 模型已存在。"
echo ""

# --- 步骤 6: 启动主程序 (并强制使用正确的代理) ---
echo "[6/6] 正在启动 Python 主程序..."

# 检查用户是否配置了代理地址
if [ -n "$PROXY_URL" ]; then
    # 如果配置了，就强制将这些环境变量设置为用户指定的正确地址
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export all_proxy="${PROXY_URL}"
    echo "✅ 代理已激活: ${PROXY_URL}"
else
    # 如果用户没配置，就清空所有代理环境变量，确保直连
    unset http_proxy
    unset https_proxy
    unset all_proxy
    echo "✅ 未配置代理，将尝试直接连接。"
fi

echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""

python3 "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束 ---
deactivate
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
echo "程序已执行完毕。"
read -p "按任意键退出..."