#!/bin/bash

# =================================================================
#    AI 招聘助理 - 启动器 (v33.1 - Gemini API 黄金版)
#    - 这是一个纯粹的 Shell 脚本，用于准备环境并启动Python程序
# =================================================================

# --- 颜色定义函数 ---
# 设置“赛博朋克·紫”主题 (亮紫色字体)
set_purple_theme() {
    echo -e "\033[95m"
    # clear # 您可以取消注释这行，每次启动时会清屏
}
# 恢复终端默认颜色
reset_colors() {
    echo -e "\033[0m"
}
# --- 颜色定义结束 ---


# --- 脚本开始时应用颜色主题 ---
set_purple_theme

# --- 关键步骤: 自动切换到脚本所在目录，确保所有相对路径都正确 ---
cd "$(dirname "$0")" || exit

# --- 配置区 ---
# 您的 Python 主程序的文件名
PYTHON_SCRIPT_NAME="21.py"

# --- 脚本主逻辑 ---
echo "========================================================"
echo "      AI 招聘助理 v33.1 - 启动器正在准备环境..."
echo "========================================================"
echo ""

# --- 步骤 1: 加载 .env 配置文件 (这是启用代理的关键) ---
echo "[1/5] 正在加载 .env 配置文件..."
if [ -f .env ]; then
  # 将 .env 文件中非注释行的变量导出为当前终端的环境变量
  export $(grep -v '^#' .env | xargs)
  echo "      > 状态: ✅ .env 配置已加载并导出。"
else
  echo "      > ❌ 严重错误: 未找到 .env 配置文件！程序无法获取API密钥和代理设置。"
  reset_colors # 退出前恢复颜色
  read -p "按任意键退出..."
  exit 1
fi
echo ""

# --- 步骤 2: 确认并应用网络代理 ---
echo "[2/5] 正在应用网络代理..."
if [ -n "$HTTP_PROXY" ]; then
    echo "      > 状态: ✅ 网络代理已根据 .env 设置为: $HTTP_PROXY"
else
    echo "      > 状态: 🟡 警告: .env 文件中未找到 HTTP_PROXY/HTTPS_PROXY 设置。"
    echo "      >      如果稍后访问 Gemini API 失败，请检查 .env 文件。"
fi
echo ""

# --- 步骤 3: 检查并准备 Python 虚拟环境 ---
echo "[3/5] 正在检查并准备 Python 虚拟环境 (venv)..."
if [ ! -d "venv" ]; then
    echo "      > 未在本目录找到虚拟环境 'venv'，正在自动创建..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "      > ❌ 严重错误: 虚拟环境创建失败！请检查您的 python3 环境。"
        reset_colors
        read -p "按任意键退出..."
        exit 1
    fi
fi
VENV_PYTHON="venv/bin/python"
PIP_EXEC="venv/bin/pip"
echo "      > 状态: ✅ 虚拟环境准备就绪。"
echo ""

# --- 步骤 4: 安装/更新 Python 依赖 ---
echo "[4/5] 正在安装/更新 Python 依赖..."
if [ ! -f "requirements.txt" ]; then
    echo "      > ❌ 严重错误: 未找到依赖文件 'requirements.txt'！"
    reset_colors
    read -p "按任意键退出..."
    exit 1
fi
# 隐藏pip的正常输出，只在出错时显示错误
${PIP_EXEC} install -r requirements.txt &> /dev/null
if [ $? -ne 0 ]; then
    echo "      > ❌ 严重错误: Python 依赖安装失败！请重新运行一次或检查 'requirements.txt' 文件。"
    # 如果失败，再次运行并显示详细输出，方便排查问题
    ${PIP_EXEC} install -r requirements.txt
    reset_colors
    read -p "按任意键退出..."
    exit 1
fi
echo "      > 状态: ✅ Python 依赖已是最新状态。"
echo ""

# --- 步骤 5: 检查并确保 Qdrant 数据库正在运行 ---
echo "[5/5] 正在检查 Qdrant 数据库 (${CENTRAL_DB_CONTAINER_NAME}) 状态..."
if [ -z "$CENTRAL_DB_CONTAINER_NAME" ]; then
    echo "      > ❌ 严重错误: .env 文件中未定义 CENTRAL_DB_CONTAINER_NAME 变量！"
    reset_colors
    read -p "按任意键退出..."
    exit 1
fi

if docker ps -q -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ | grep -q .; then
    echo "      > 状态: ✅ Qdrant 数据库正在运行。"
elif docker ps -aq -f name=^/${CENTRAL_DB_CONTAINER_NAME}$ | grep -q .; then
    echo "      > 检测到已停止的 Qdrant 容器，正在尝试启动..."
    docker start ${CENTRAL_DB_CONTAINER_NAME} > /dev/null
    echo "      > 状态: ✅ Qdrant 数据库已成功启动。"
else
    echo "      > ❌ 严重错误: 未找到名为 '${CENTRAL_DB_CONTAINER_NAME}' 的 Docker 容器！"
    reset_colors
    read -p "按任意键退出..."
    exit 1
fi
echo ""

# --- 最后一步: 启动 Python 主程序 ---
echo ">>> 正在启动 Python 主程序 (${PYTHON_SCRIPT_NAME})..."
echo "-------------------------- [ 程序日志开始 ] --------------------------"
echo ""

"${VENV_PYTHON}" "${PYTHON_SCRIPT_NAME}"

# --- 脚本结束后的提示 ---
echo ""
echo "-------------------------- [ 程序日志结束 ] --------------------------"
echo "程序已结束。如果您想再次运行，请重新执行脚本。"
reset_colors
read -p "按任意键关闭此终端..."