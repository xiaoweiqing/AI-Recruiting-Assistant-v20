#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===================================================================================
# AI 招聘助理 v33.1 - 增强结构化数据提取与彩色输出
# ===================================================================================
# 版本说明:
# - 【【【 核心升级：增强结构化 & 彩色高亮 】】】:
#   1.  【数据库增强】: 新增 电话(phone), 邮箱(email), 在职状态(status), 核心匹配点(core_strengths),
#                     核心风险点(core_gaps), AI思考过程(ai_thinking_process) 等字段，实现更精细的数据存储。
#   2.  【AI指令优化】: 更新Prompt，引导AI直接在JSON中输出更多结构化信息。
#   3.  【关键信息提取】: 通过解析AI的最终输出来自动填充新增的数据库字段。
#   4.  【彩色高亮】: 在命令行终端，使用醒目的颜色高亮显示 "核心匹配点" (绿色) 和 "核心风险点" (黄色)，使其一目了然。
#   5.  【继承优点】: 完整保留了v33.0的流式输出、数据隔离等全部核心功能。
# ===================================================================================
# --- 核心 import ---
import asyncio, json, os, sys, platform, re, shutil, sqlite3, subprocess, threading, time, traceback, unicodedata, uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

# --- 【【【 在这里添加 FastAPI 相关的 import 】】】 ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

# --- 【【【 添加结束 】】】 ---

# --- LangChain & AI 相关的导入 ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 【【【 在这里添加下面三行 】】】
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# 【【【 添加结束 】】】
from numpy import dot
from numpy.linalg import norm

# ===================================================================
# --- 【【【 从这里开始，完整复制下面的新代码块 】】】 ---
# ===================================================================

# --- 1. Qdrant 相关的 import ---
try:
    from qdrant_client import QdrantClient, models

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print(">> [Warning] Qdrant client not found. Vectorization will be disabled.")
    print("   -> Please run: pip install qdrant-client")

# --- 2. Qdrant 相关的全局变量 ---
QDRANT_CLIENT: QdrantClient = None
QDRANT_COLLECTION_NAME = "ai_recruiter_v33_resumes"  # 独立的向量数据库名称

# --- 3. 汇总报告相关的全局变量 ---
SUMMARY_REPORT_DIR = Path(__file__).resolve().parent / "所有汇总报告"  # 报告保存路径
all_session_results = []  # 用来暂存本次运行的所有分析结果
session_results_lock = asyncio.Lock()  # 保证多任务同时写入列表时数据安全

# ===================================================================
# --- 【【【 复制到这里结束 】】】 ---
# ===================================================================
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print(
        ">> [Warning] Pillow or Pytesseract not found. Image processing will be disabled."
    )


# --- 全局配置与类 ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"


# ... (其他 import)

load_dotenv()

# --- 【【【 核心修改：定义专属的项目内监控文件夹 】】】 ---
# 获取当前脚本所在的目录
PROJECT_ROOT_DIR = Path(__file__).resolve().parent
# 在项目根目录下创建一个名为 "AI_Recruiter_IPC" 的文件夹来存放所有交互文件
IPC_DIR = PROJECT_ROOT_DIR / "AI_Recruiter_IPC"
# 定义新的 inbox 和 archive 文件夹
INBOX_DIR = IPC_DIR / "inbox"
ARCHIVE_DIR = IPC_DIR / "archive"
# --- 【【【 修改结束 】】】 ---

JDS_FOLDER_PATH = Path(__file__).resolve().parent / "JDs_library"

# ... (脚本其余部分)

# --- [NEW] 数据库文件名保持不变，通过代码动态添加新列 ---
LOCAL_DATABASE_FILE = IPC_DIR / "recruitment_data_v33_streaming.db"

# --- [MODIFIED] AI 指令 (Prompt) 更新 ---
# --- [MODIFIED] AI 指令 (Prompt) 更新 ---
AI_PROMPT_TEMPLATE = """### INSTRUCTION:
You are a top-tier AI Technical Recruiter, hiring for a deeply technical AI Architect/Director role. Your primary mission is to identify candidates with world-class engineering and architectural skills.

**### Evaluation Criteria & Weighting (VERY IMPORTANT):**
This is a technology-first role. You MUST evaluate candidates based on the following weights:
- **70% Technical Prowess:** This is the most critical part. Assess their hands-on experience with LLM deployment/fine-tuning, system architecture, Python proficiency, and the specified tech stack (OCR, Vector DBs). A major gap in the "Core Technical Skills" section of the JD is a deal-breaker and must result in a significantly lower score.
- **30% Management & Business Acumen:** This includes team leadership, project management, and client communication. These skills are important for applying the technology, but they CANNOT compensate for technical weakness.

**1. Structured Data Extraction (JSON):**
First, extract the key information from the resume and format it as a JSON object. This block MUST start with ```json and end with ```. If a value is not found, use "N/A".

```json
{{
  "candidate_name": "The candidate's full name, or 'Unknown' if not found",
  "phone": "The candidate's phone number, or 'N/A'",
  "email": "The candidate's email address, or 'N/A'",
  "status": "The candidate's current job status ('on-the-job' or 'searching'), or 'N/A'",
  # ... (前面省略)
  "final_match_score_percent": "Your final, ADJUSTED score as a number only, after applying the 70/30 weighting rule."
}}
2. Stream of Consciousness (English):
Next, think step-by-step in English. 
- First, acknowledge the initial vector score as a starting reference.
- Then, critically compare the resume against the job's hard requirements (like years of experience, team size, domain knowledge).
- Based on any major gaps you find, explicitly state how you will adjust the score. For example: "The vector score is 78%, but the lack of required management experience is a critical gap. Therefore, I will adjust the final score down to 65%."
This part is your detailed thought process for adjusting the score.
# ...
3. Final Verdict (Bilingual Summary):
Finally, you MUST conclude with a final, clearly formatted summary. This summary is the only part that needs to be bilingual.

**岗位向量匹配度详情 (Vector Match Details):**
| 岗位名称 (Job Title) | 匹配度 (Match Score) |
|--------------------|----------------------|
| [请在此处填入'业务分析师'的向量匹配度] | [例如: 63.9%] |
| [请在此处填入'python'的向量匹配度] | [例如: 63.0%] |
| [请在此处填入'测试经理'的向量匹配度] | [例如: 61.7%] |
*This table MUST reflect the vector scores from Stage 1.*

[ AI 总评 / FINAL VERDICT ]
核心匹配点 (Core Strengths):
# ...
[用中文总结最强的1-2个匹配点]
(A brief English summary of the strongest matching points)
核心风险点 (Potential Gaps):
[用中文总结最主要的1-2个风险或疑问]
# ... (前面省略)
(A brief English summary of the main risks or questions)
Final Match Score: XX%
(This score MUST be your final, adjusted score based on your analysis, not the initial vector score.)
JOB DESCRIPTIONS:
# ...
{jd_input}
CANDIDATE RESUME:
{resume_text}
YOUR FULL RESPONSE (JSON, Analysis, and Verdict):
"""
# --- 全局变量 ---
llm, ACTIVE_JD_DATA, app_controller = None, {}, None
# --- 在这里添加下面这行 ---
HIGH_PERFORMERS_DIR = Path(__file__).resolve().parent / "绩优候选人简历"
# --- 【【【 在这里添加下面两行代码 】】】 ---
COMPARISON_PARTICIPANTS_DIR = Path(__file__).resolve().parent / "参与对比的候选人简历"
POTENTIAL_CANDIDATES_DIR = Path(__file__).resolve().parent / "潜力候选人简历"
# --- 【【【 在这里添加下面这行 】】】 ---
RANKED_RESUMES_DIR = Path(__file__).resolve().parent / "已分析简历存档"
# --- 【【【 添加结束 】】】 ---
# --- 【【【 添加结束 】】】 ---
# --- 【【【 第1步：添加以下全局变量 】】】 ---
SUMMARY_REPORT_DIR = Path(__file__).resolve().parent / "所有汇总报告"  # 报告保存路径
all_session_results = []  # 用来暂存本次运行的所有分析结果
session_results_lock = asyncio.Lock()  # 保证多任务同时写入列表时数据安全
EMBEDDING_MODEL: HuggingFaceEmbeddings = None
task_queue = asyncio.Queue()
candidate_id_counter = None

# ===================================================================================
# --- 【【【 V2 新增：对比系统专属全局变量与AI指令 】】】 ---
# ===================================================================================

# --- 1. 对比任务专属全局变量 ---
QDRANT_COMPARISON_COLLECTION_NAME = (
    "ai_recruiter_v33_comparisons"  # 对比报告的独立向量库
)
all_comparison_results = []  # 用来暂存本次运行的所有对比任务结果
# --- 【【【 V3 新增：在这里添加下面这行 】】】 ---
ACTIVE_COMPARISON_TASK = None  # 用于存放预加载的对比任务模板
# --- 2. 对比任务专属AI指令 (Prompt) ---
PROMPT_CANDIDATE_COMPARISON = """
# 角色
你是一位拥有15年经验的资深技术招聘专家，任务是针对一个具体的【{job_title}】职位，对两位候选人进行精准的横向对比，并以结构化的JSON和详细的分析文本两种形式输出。

# 输入材料
1.  **【职位要求 (JD)】**
    ```    {jd_text}
    ```
2.  **【基准候选人简历 (Benchmark Candidate)】**
    - 姓名: {benchmark_name}
    ```
    {benchmark_resume_text}
    ```
3.  **【新候选人简历 (New Candidate)】**
    - 姓名: {new_candidate_name}
    ```
    {new_resume_text}
    ```

# 任务指令
必须严格按照以下两步输出：

**第一部分: 结构化数据提取 (JSON)**
首先，生成一个包含你核心判断的JSON对象。这个块必须以 ```json 开始，以 ``` 结束。所有分析都必须围绕提供的JD进行。

```json
{{
  "benchmark_candidate_score": "为基准候选人打一个0-100的综合分",
  "new_candidate_score": "为新候选人打一个0-100的综合分",
  "verdict": "明确指出谁更胜出 ('新候选人胜出', '基准候选人胜出', 或 '综合实力相当')",
  "new_candidate_win_points": [
    "新候选人最明显的第一个优势",
    "新候选人最明显的第二个优势"
  ],
  "new_candidate_risk_points": [
    "新候选人最明显的第一个劣势或风险点"
  ]
}}
**第二部分: 详细分析报告 (Markdown)**
在JSON块之后，生成一份详细的、具有深度洞察的分析报告。使用Markdown格式，内容包括：

- **【综合评分对比】**: 分别陈述两位候选人的得分和打分依据。
- **【新候选人 vs. 基准候选人：优势 (Win Points)】**: 详细阐述“新候选人”明显优于“基准候选人”的3-5个关键点，并结合JD说明原因。
- **【新候选人 vs. 基准候选人：劣势 (Risk Points)】**: 详细阐述“新候选人”可能不如“基准候选人”的2-3个潜在风险点或不确定性。
- **【最终结论与建议】**: 基于以上所有分析，用一段话总结你的最终推荐建议。明确回答：“新候选人是否比基准候选人更值得推荐？”，并给出核心理由。
"""

PROMPT_DEEP_DIVE_JD_MATCH = """
# 角色
你是一位专业的AI招聘分析师，任务是深度剖析一份候选人简历与职位要求的匹配度。

# 输入材料
1.  **【职位要求 (JD)】**
    ```    {jd_text}
    ```
2.  **【候选人简历】**
    ```
    {resume_text}
    ```

# 任务指令
请生成一份关于这位候选人与职位匹配度的详细分析报告，包含以下部分：

1.  **【总体匹配度评分】**:
    - 请给出一个0-100分之间的分数，并附上一句精炼的总体评价。

2.  **【核心匹配点 (Strengths)】**:
    - 详细列出候选人的经历、技能或项目中最符合JD要求的3-5个关键点。

3.  **【潜在风险与不匹配点 (Weaknesses/Gaps)】**:
    - 客观地指出简历中未能体现或可能不满足JD要求的2-3个方面。

4.  **【面试建议问题 (Interview Questions)】**:
    - 基于发现的潜在风险点，提出3个有针对性的面试问题，用于在面试中进一步考察候选人。
"""
# ===================================================================================
# --- 【【【 V2 新增结束 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 初始化与设置模块 ---
# ===================================================================================


# --- [MODIFIED] 完整替换此函数 ---
# --- [MODIFIED V2] 完整替换此函数 ---
def setup_local_database():
    """初始化数据库，并动态为【单简历分析】和【对比分析】创建和更新表结构。"""
    print(">> [DB] Initializing SQLite database...")
    try:
        with sqlite3.connect(LOCAL_DATABASE_FILE) as conn:
            cursor = conn.cursor()

            # --- 模块1: 单简历分析表 (reports) ---
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY, total_id INTEGER UNIQUE, candidate_name TEXT, vector_score TEXT,
                final_match_score TEXT, ai_analysis TEXT, raw_resume TEXT, creation_time TEXT )
            """
            )
            table_info = cursor.execute("PRAGMA table_info(reports)").fetchall()
            existing_columns = [col[1] for col in table_info]
            new_columns = {
                "phone": "TEXT",
                "email": "TEXT",
                "status": "TEXT",
                "core_strengths": "TEXT",
                "core_gaps": "TEXT",
                "ai_thinking_process": "TEXT",
            }
            for col_name, col_type in new_columns.items():
                if col_name not in existing_columns:
                    print(f"   -> [DB] Adding column to 'reports': '{col_name}'...")
                    cursor.execute(
                        f"ALTER TABLE reports ADD COLUMN {col_name} {col_type}"
                    )

            # --- 模块2: 对比分析表 (comparisons) ---
            print("   -> [DB] Checking 'comparisons' table...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT,
                jd_name TEXT,
                benchmark_candidate_name TEXT,
                new_candidate_name TEXT,
                benchmark_candidate_score INTEGER,
                new_candidate_score INTEGER,
                verdict TEXT,
                full_pk_report TEXT,
                full_jd_match_report TEXT,
                creation_time TEXT
            )
            """
            )

        print(f"✅ [DB] Database '{LOCAL_DATABASE_FILE.name}' is up to date.")
        return True
    except Exception as e:
        print(f"❌ [DB] FATAL ERROR: {e}")
        return False


# ===================================================================================
# --- 【【【 Gemini API 版本：用这个新函数，完整替换旧的 setup_api_and_embedder 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 最终修正版：用这个新函数，完整替换旧的 setup_api_and_embedder 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 最终解决方案：用这个函数完整替换旧的 setup_api_and_embedder 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 双保险最终版：用这个函数完整替换旧的 setup_api_and_embedder 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 请使用这个【【【调试专用版】】】】来替换您现有的 setup_api_and_embedder 函数 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 最终解决方案：用这个函数完整替换旧的 setup_api_and_embedder 】】】 ---
# ===================================================================================
def setup_api_and_embedder():
    """
    初始化 Google Gemini API 用于分析，并从本地加载 HuggingFace 模型用于向量化。
    【【最终修正版】】: 修正了错误的 embedding 模型名称。
    """
    global llm, EMBEDDING_MODEL

    # --- 第一部分: 初始化 Google Gemini API (这部分已经验证是成功的，保持不变) ---
    try:
        gemini_api_key = os.getenv("API_KEY")
        if not gemini_api_key:
            print(
                f"❌ {Colors.RED}[AI] 致命错误: 未在 .env 文件中找到 'API_KEY'。{Colors.RESET}"
            )
            return False

        print(f">> [AI] 正在连接至 Google Gemini API (使用代理友好的 REST 模式)...")
        genai.configure(api_key=gemini_api_key, transport="rest")

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0.1,
            safety_settings=safety_settings,
            google_api_key=gemini_api_key,
        )
        llm.invoke("Hi")
        print(
            f"✅ {Colors.GREEN}[AI] 连接 Gemini API 成功！模型: gemini-1.5-flash-latest{Colors.RESET}"
        )

    except Exception as e:
        print(
            f"❌ {Colors.RED}[AI] 致命错误: 初始化 Gemini 模型失败: {e}{Colors.RESET}"
        )
        traceback.print_exc()
        return False

    # --- 第二部分: 初始化本地 Embedding 模型 (【【【这里是修正点】】】) ---
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        # 修正了模型名称从 "embeddinggemma-2b" 到 "embeddinggemma-300m"
        model_name_to_load = "google/embeddinggemma-300m"
        print(
            f">> [Embedder] 正在从本地缓存加载向量化模型 '{model_name_to_load}' (离线模式)..."
        )
        EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=model_name_to_load,
            model_kwargs={"device": "cpu"},
        )
        print(f"✅ {Colors.GREEN}[Embedder] 向量化模型已从缓存加载成功。{Colors.RESET}")
    except Exception as e:
        print(f"❌ {Colors.RED}[Embedder] 致命错误: {e}{Colors.RESET}")
        traceback.print_exc()
        return False

    return True


# --- [MODIFIED] 完整替换此函数 ---
def load_and_vectorize_jds():
    global ACTIVE_JD_DATA
    print(f">> [JD] Syncing & vectorizing JDs from {JDS_FOLDER_PATH}")
    JDS_FOLDER_PATH.mkdir(exist_ok=True)

    # --- 【修改点 1/3】: 更新用户提示信息 ---
    print(
        f"{Colors.CYAN}>> [JD Filter] 提示: 要临时禁用某个JD, 只需在其文件名前添加减号 '-' (例如: '-python.txt'){Colors.RESET}"
    )

    all_jd_files = list(JDS_FOLDER_PATH.glob("*.txt"))
    # --- 【修改点 2/3】: 将 '_' 修改为 '-' ---
    active_jd_files = [fp for fp in all_jd_files if not fp.name.startswith("-")]
    # --- 【修改点 3/3】: 将 '_' 修改为 '-' ---
    inactive_jd_files = [fp.name for fp in all_jd_files if fp.name.startswith("-")]

    jds = {}
    for fp in active_jd_files:
        try:
            content = fp.read_text("utf-8")
            if content.strip():
                jds[fp.stem.lstrip("-")] = {
                    "content": content,
                    "vector": EMBEDDING_MODEL.embed_query(content),
                }
            else:
                print(f" 🟡 [JD] Warning: '{fp.name}' is empty and will be skipped.")
        except Exception as e:
            print(f" ❌ [JD] Failed to process '{fp.name}': {e}")

    if not jds:
        print(
            f"{Colors.YELLOW}!! [JD] Warning: No active JDs found to load.{Colors.RESET}"
        )

    ACTIVE_JD_DATA = jds

    print(
        f"✅ [JD] Loaded and vectorized {Colors.BOLD}{len(jds)}{Colors.RESET} active JDs."
    )
    if inactive_jd_files:
        print(
            f"   -> Ignored {len(inactive_jd_files)} inactive JDs: {', '.join(inactive_jd_files)}"
        )
    return True


def get_current_max_total_id():
    try:
        with sqlite3.connect(LOCAL_DATABASE_FILE) as conn:
            res = (
                conn.cursor().execute("SELECT MAX(total_id) FROM reports").fetchone()[0]
            )
            return res or 0
    except:
        return 0


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


# ===================================================================================
# --- 【【【 在这里粘贴新的通知函数 】】】 ---
# ===================================================================================
def safe_notification(title, message):
    """
    使用 notify-send 命令发送桌面通知，并提供详细的调试信息。
    """
    print(f"\n   -> {Colors.MAGENTA}[通知] 正在发送桌面通知...{Colors.RESET}")
    try:
        result = subprocess.run(
            ["notify-send", title, message, "-a", "AI Recruiter", "-t", "8000"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(
            f"   ✅ {Colors.MAGENTA}[通知] 'notify-send' 命令已成功触发。{Colors.RESET}"
        )
    except FileNotFoundError:
        print(
            f"   !! {Colors.RED}[通知] 严重错误: 'notify-send' 命令未找到。{Colors.RESET}"
        )
    except subprocess.CalledProcessError as e:
        print(
            f"   !! {Colors.RED}[通知] 'notify-send' 执行失败: {e.stderr.strip()}{Colors.RESET}"
        )
    except Exception as e:
        print(f"   !! {Colors.RED}[通知] 发送通知时发生未知错误: {e}{Colors.RESET}")


# ===================================================================================
# --- 【【【 V2 新增：对比分析核心引擎 (共6个函数) 】】】 ---
# ===================================================================================


async def call_llm_for_analysis_async(prompt_text: str, worker_name: str) -> str:
    """一个通用的、用于调用LLM进行分析并流式输出的异步函数。"""
    color_map = {"PK-Analyst": Colors.CYAN, "JD-Matcher": Colors.MAGENTA}
    color = color_map.get(worker_name, Colors.CYAN)
    print(
        f"\n{color}>> [{worker_name}] 正在请求AI进行深度分析... (实时流式输出){Colors.RESET}"
    )

    full_response = ""
    try:
        async for chunk in llm.astream(prompt_text):
            content_chunk = chunk.content
            print(content_chunk, end="", flush=True)
            full_response += content_chunk
        print(f"\n{color}✅ [{worker_name}] AI分析完成。{Colors.RESET}")
        return full_response
    except Exception as e:
        print(f"\n{Colors.RED}!! [Model Error during {worker_name}] {e}{Colors.RESET}")
        return f"AI分析失败: {e}"


def read_text_file_safely(file_path: Path) -> str:
    """安全地读取文本文件内容。"""
    try:
        return file_path.read_text("utf-8")
    except Exception as e:
        print(
            f"  {Colors.RED}!! 读取文件失败: {file_path.name}, 错误: {e}{Colors.RESET}"
        )
        return None


def parse_candidate_name(filename: str) -> str:
    """从文件名中提取候选人姓名。"""
    # 移除关键字和扩展名，然后将下划线替换为空格
    base = Path(filename).stem
    name = base.replace("benchmark", "").replace("new_candidate", "").strip(" _-")
    return name.replace("_", " ")


# ===================================================================================
# --- 【【【 在这里粘贴修正后的函数 】】】 ---
# ===================================================================================


async def background_comparison_storage_task(task_data: dict):
    """【后台任务】将对比分析结果存入专属的SQLite表。"""
    try:
        with sqlite3.connect(LOCAL_DATABASE_FILE) as conn:
            query = """
            INSERT INTO comparisons (task_name, jd_name, benchmark_candidate_name, new_candidate_name,
                                     benchmark_candidate_score, new_candidate_score, verdict,
                                     full_pk_report, full_jd_match_report, creation_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                task_data["task_name"],
                task_data["jd_name"],
                task_data["benchmark_name"],
                task_data["new_candidate_name"],
                task_data.get("benchmark_score"),
                task_data.get("new_candidate_score"),
                task_data.get("verdict"),
                task_data.get("pk_report", "N/A"),  # <-- 也改为.get()更安全
                # --- 【【【 核心修正点 】】】 ---
                # 将强制的 [] 访问改为安全的 .get() 访问，提供一个默认值
                task_data.get("jd_match_report", "N/A"),
                # --- 【【【 修正结束 】】】 ---
                datetime.now(timezone(timedelta(hours=8))).isoformat(),
            )
            conn.execute(query, params)
            print(f" > [Sync] Comparison '{task_data['task_name']}' saved to SQLite.")
    except Exception as e:
        print(
            f"{Colors.RED}!! [Sync Error] SQLite write failed for comparison '{task_data['task_name']}': {e}{Colors.RESET}"
        )


# ===================================================================================
# --- 【【【 粘贴到这里结束 】】】 ---
# ===================================================================================```


async def vectorize_and_store_comparison_async(task_data: dict):
    """【后台任务】将完整的对比报告向量化并存入Qdrant。"""
    if not QDRANT_CLIENT:
        return

    full_report_text = f"对比任务: {task_data['task_name']}\n\nPK报告:\n{task_data['pk_report']}\n\nJD匹配报告:\n{task_data['jd_match_report']}"
    try:
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(
            None, EMBEDDING_MODEL.embed_query, full_report_text
        )

        upsert_op = lambda: QDRANT_CLIENT.upsert(
            collection_name=QDRANT_COMPARISON_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "task_name": task_data["task_name"],
                        "verdict": task_data.get("verdict"),
                        "new_candidate_name": task_data["new_candidate_name"],
                        "text_snippet": full_report_text[:500],
                    },
                )
            ],
            wait=True,
        )
        await loop.run_in_executor(None, upsert_op)
        print(
            f" > [Qdrant Sync] Comparison '{task_data['task_name']}' report vectorized."
        )
    except Exception as e:
        print(
            f"{Colors.RED}!! [Qdrant Sync Error] Vectorization failed for '{task_data['task_name']}': {e}{Colors.RESET}"
        )


# ===================================================================================
# --- 【【【 V2 新增结束 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 核心分析、存储与展示模块 ---
# ===================================================================================
def pre_extract_contact_info(text: str) -> dict:
    """使用正则表达式预先提取联系信息，作为AI的补充。"""
    info = {"phone": "N/A", "email": "N/A"}

    try:
        # 移除了所有空格和破折号，以匹配如 '135-1234-5678' 或 '135 1234 5678' 的格式
        cleaned_text_for_phone = text.replace(" ", "").replace("-", "")
        phone_match = re.search(r"1[3-9]\d{9}", cleaned_text_for_phone)
        if phone_match:
            info["phone"] = phone_match.group(0)

        # 邮箱的正则表达式
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        if email_match:
            info["email"] = email_match.group(0)
    except Exception as e:
        print(f"!! [Regex Pre-extract Error] {e}")

    return info


def vector_similarity_analysis(resume_text: str):
    print("\n" + "=" * 25 + " Stage 1: Vector Similarity Quick Scan " + "=" * 25)
    if not ACTIVE_JD_DATA:
        print("No JDs loaded to compare against.")
        return {}

    resume_vector = EMBEDDING_MODEL.embed_query(resume_text)
    scores = {}
    for title, data in ACTIVE_JD_DATA.items():
        similarity = dot(resume_vector, data["vector"]) / (
            norm(resume_vector) * norm(data["vector"])
        )
        scores[title] = max(0, min(100, similarity * 100))

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    for title, score in sorted_scores:
        blocks = int(score / 4)
        color = (
            Colors.GREEN if score > 75 else Colors.YELLOW if score > 50 else Colors.CYAN
        )
        print(
            f"{title:<40} | {color}{'█' * blocks}{Colors.RESET}{' ' * (25 - blocks)} | {score:.1f}%"
        )
    print("=" * 80)

    return {t: f"{s:.1f}%" for t, s in sorted_scores}


# --- [MODIFIED] 完整替换此函数 ---
# --- [MODIFIED] 完整替换此函数 ---
# --- 【【【 V3 MODIFIED: 将此函数改为 async def 并整合所有逻辑 】】】 ---
# --- 【【【 V3.2 MODIFIED: 替换此函数以修正bug 】】】 ---
# --- 【【【 V3.7 MODIFIED: 替换此函数以实现动态JD表格 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 第1步：在这里粘贴全新的“文件名净化”函数 】】】 ---
# ===================================================================================
def sanitize_filename(name: str) -> str:
    """
    移除或替换字符串中的非法字符，使其可以安全地用作文件名。
    """
    if not isinstance(name, str):
        name = str(name)
    # 将最常见的路径分隔符替换为下划线
    name = name.replace("/", "_").replace("\\", "_")
    # 移除其他所有Windows和Linux不允许的字符
    return re.sub(r'[*:?"<>|]', "", name).strip()


# ===================================================================================


# ... (这个函数本身不用动)
# ===================================================================================
# --- 【【【 第三步：添加这个全新的姓名提取辅助函数 】】】 ---
# ===================================================================================
def get_name_with_fallback(ai_name: str, resume_text: str, filename: str = None) -> str:
    """
    智能提取候选人姓名，如果AI未能识别，则尝试从简历首字或文件名中提取姓氏作为备用。
    """
    # 1. 优先使用AI提取的有效名称
    if ai_name and ai_name.lower() not in ["unknown", "n/a", ""]:
        # 移除可能存在的 "**"
        return ai_name.replace("*", "").strip()

    # 2. 备用方案：尝试从简历文本的第一个字提取姓氏
    # 检查文本是否以中文字符开头
    if resume_text and "\u4e00" <= resume_text[0] <= "\u9fff":
        surname = resume_text[0]
        return f"{surname}先生/女士"

    # 3. 最后备用方案：从文件名解析
    if filename:
        # 使用正则表达式匹配开头的中文字符作为姓氏
        match = re.match(r"^[\u4e00-\u9fa5]+", Path(filename).stem)
        if match:
            surname = match.group(0)
            # 如果文件名就是 "宋**简历.txt"，这里会提取 "宋"
            return f"{surname.replace('*','')}先生/女士"

    # 4. 如果所有方法都失败，返回一个明确的提示
    return "姓名无法识别"


# ===================================================================================
# ===================================================================================
# --- 【【【 第三步：用这个功能最完整的版本，替换现有函数 】】】 ---
# ===================================================================================
# ===================================================================================


# --- 【【【 使用这个修正版，完整替换您现有的同名函数 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 第2步：用这个最终修正版，完整替换您现有的同名函数 】】】 ---
# ===================================================================================
async def trigger_smart_analysis_async(content, new_id):
    resume_text = ""
    original_filename = None
    original_file_path = None

    if isinstance(content, Path):
        original_filename = content.name
        original_file_path = content
        try:
            if PIL_AVAILABLE and content.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                img = Image.open(str(content))
                resume_text = clean_text(
                    pytesseract.image_to_string(img, "chi_sim+eng")
                )
            elif content.suffix.lower() in [".txt", ".md", ".rtf"]:
                resume_text = clean_text(content.read_text("utf-8"))
            else:
                print(
                    f" -> 🟡 Unsupported file type for content extraction: {content.suffix}"
                )
        except Exception as e:
            print(
                f"!! [File Read/OCR Error] Failed to process file {original_filename}: {e}"
            )
            resume_text = ""
        finally:
            if original_file_path and original_file_path.exists():
                shutil.move(str(original_file_path), ARCHIVE_DIR / original_filename)
    else:
        resume_text = clean_text(content)

    if not resume_text:
        print(f">> [Pre-check] ID {new_id}: Content is empty, skipped.")
        return

    # --- Stage 1 & 2: Vector Scan & AI Analysis ---
    vector_scores = vector_similarity_analysis(resume_text)
    jd_input_text = "\n\n---\n\n".join(
        [f"## {t}\n{d['content']}" for t, d in ACTIVE_JD_DATA.items()]
    )
    dynamic_table_header = "| 岗位名称 (Job Title) | 匹配度 (Match Score) |\n|--------------------|----------------------|"
    dynamic_table_rows = [
        f"| {title} | [请在此处填入匹配度] |" for title in ACTIVE_JD_DATA.keys()
    ]
    dynamic_vector_table_instruction = f"{dynamic_table_header}\n" + "\n".join(
        dynamic_table_rows
    )
    modified_template = AI_PROMPT_TEMPLATE.replace(
        "| 岗位名称 (Job Title) | 匹配度 (Match Score) |\n|--------------------|----------------------|\n| [请在此处填入'业务分析师'的向量匹配度] | [例如: 63.9%] |\n| [请在此处填入'python'的向量匹配度] | [例如: 63.0%] |\n| [请在此处填入'测试经理'的向量匹配度] | [例如: 61.7%] |",
        dynamic_vector_table_instruction,
    )
    prompt = modified_template.format(jd_input=jd_input_text, resume_text=resume_text)

    print(
        "\n" + "=" * 28 + f" Stage 2: AI In-depth Analysis [ID: {new_id}] " + "=" * 28
    )
    full_response = await call_llm_for_analysis_async(prompt, "Standard-Analyst")

    # --- Stage 3: Parsing and Data Cleanup ---
    structured_data, ai_thinking_process = {}, "Parsing failed."
    try:
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", full_response, re.DOTALL)
        if json_match:
            structured_data = json.loads(json_match.group(1))
            verdict_start_index = full_response.find("[ AI 总评 / FINAL VERDICT ]")
            ai_thinking_process = full_response[
                len(json_match.group(0)) : verdict_start_index
            ].strip()
    except Exception as e:
        ai_thinking_process = f"Error during JSON parsing: {e}"

    candidate_name = get_name_with_fallback(
        structured_data.get("candidate_name"), resume_text, original_filename
    )
    strengths_match = re.search(
        r"核心匹配点 \(Core Strengths\):\s*(.*?)\s*核心风险点", full_response, re.DOTALL
    )
    gaps_match = re.search(
        r"核心风险点 \(Potential Gaps\):\s*(.*?)\s*Final Match Score:",
        full_response,
        re.DOTALL,
    )
    core_strengths = strengths_match.group(1).strip() if strengths_match else "未能提取"
    core_gaps = gaps_match.group(1).strip() if gaps_match else "未能提取"

    score_text = structured_data.get("final_match_score_percent", "N/A")
    try:
        final_score_value = float(re.sub(r"[^0-9.]", "", str(score_text)))
    except (ValueError, TypeError):
        final_score_value = -1
    final_score_str = f"{score_text}%" if final_score_value != -1 else "N/A"

    print("\n" + "=" * 80)
    print(f"✅【AI VERDICT】Candidate: {candidate_name} [ID: {new_id}]")
    print(f"{Colors.GREEN}{Colors.BOLD}核心匹配点:{Colors.RESET}\n{core_strengths}")
    print(f"{Colors.YELLOW}{Colors.BOLD}核心风险点:{Colors.RESET}\n{core_gaps}")
    print(f"Final Score: {Colors.BOLD}{Colors.MAGENTA}{final_score_str}{Colors.RESET}")
    print("=" * 80)

    # --- Stage 4: Background Storage (for all candidates) ---
    summary_data = {
        "id": new_id,
        "name": candidate_name,
        "score": score_text,
        "strengths": core_strengths,
        "gaps": core_gaps,
        "vector_scores": vector_scores,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_resume": resume_text,
    }
    async with session_results_lock:
        all_session_results.append(summary_data)
    asyncio.create_task(
        background_storage_task(
            new_id,
            candidate_name,
            structured_data.get("phone", "N/A"),
            structured_data.get("email", "N/A"),
            structured_data.get("status", "N/A"),
            json.dumps(vector_scores, ensure_ascii=False),
            final_score_str,
            core_strengths,
            core_gaps,
            ai_thinking_process,
            full_response,
            resume_text,
        )
    )
    if resume_text:
        asyncio.create_task(
            vectorize_and_store_resume_async(new_id, candidate_name, resume_text)
        )

    # --- Stage 5: 智能决策与三级归档系统 ---
    comparison_happened = False

    # 【【【 核心修正点：在这里统一调用净化函数 】】】
    safe_candidate_name = sanitize_filename(candidate_name)
    safe_benchmark_name = (
        sanitize_filename(ACTIVE_COMPARISON_TASK["benchmark_name"])
        if ACTIVE_COMPARISON_TASK
        else ""
    )

    # --- 决策 1: 是否进行对比分析 (分数 >= 70) ---
    if ACTIVE_COMPARISON_TASK and final_score_value >= 70:
        comparison_happened = True
        print(
            f"\n{Colors.CYAN}>> [Decision] Score ({final_score_value}%) >= 70%. Triggering comparison analysis...{Colors.RESET}"
        )

        pk_prompt = PROMPT_CANDIDATE_COMPARISON.format(
            job_title=ACTIVE_COMPARISON_TASK["task_name"],
            jd_text=ACTIVE_COMPARISON_TASK["jd_text"],
            benchmark_resume_text=ACTIVE_COMPARISON_TASK["benchmark_resume_text"],
            benchmark_name=ACTIVE_COMPARISON_TASK["benchmark_name"],
            new_resume_text=resume_text,
            new_candidate_name=candidate_name,
        )
        pk_full_response = await call_llm_for_analysis_async(pk_prompt, "PK-Analyst")

        pk_json_data = {}
        try:
            pk_json_match = re.search(
                r"```json\s*(.*?)\s*```", pk_full_response, re.DOTALL
            )
            if pk_json_match:
                pk_json_data = json.loads(pk_json_match.group(1))
        except Exception as e:
            print(f"!! [PK JSON Parse Error] ID {new_id}: {e}")

        task_data_for_storage = {
            "task_name": f"{ACTIVE_COMPARISON_TASK['task_name']}_vs_{candidate_name}",
            "jd_name": ACTIVE_COMPARISON_TASK.get("jd_name", "N/A"),
            "benchmark_name": ACTIVE_COMPARISON_TASK["benchmark_name"],
            "new_candidate_name": candidate_name,
            "benchmark_score": int(pk_json_data.get("benchmark_score", -1)),
            "new_candidate_score": int(pk_json_data.get("new_candidate_score", -1)),
            "verdict": pk_json_data.get("verdict", "结论未知"),
            "pk_report": (
                pk_full_response[pk_json_match.end() :].strip()
                if "pk_json_match" in locals() and pk_json_match
                else pk_full_response
            ),
        }
        asyncio.create_task(background_comparison_storage_task(task_data_for_storage))
        async with session_results_lock:
            all_comparison_results.append(task_data_for_storage)

        # --- 【【【 归档 1: 参与对比者 (只要对比就归档) 】】】 ---
        print(
            f"{Colors.CYAN}>> [归档] 正在保存【参与对比】的候选人: {candidate_name}...{Colors.RESET}"
        )
        dest_filename = f"{safe_candidate_name}_{new_id}_{final_score_str}_vs_{safe_benchmark_name}.txt"
        dest_path = COMPARISON_PARTICIPANTS_DIR / dest_filename
        dest_path.write_text(resume_text, encoding="utf-8")
        print(f"   -> ✅ 简历已保存至: {dest_path.parent.name}/{dest_path.name}")

        # --- 【【【 归档 2: 绩优生 (作为对比者的子集，只有胜利才归档) 】】】 ---
        verdict = pk_json_data.get("verdict", "")
        if "新候选人胜出" in verdict:
            print(
                f"{Colors.GREEN}>> [归档] 候选人胜出！正在归档至【绩优候选人简历】...{Colors.RESET}"
            )
            new_candidate_score = int(pk_json_data.get("new_candidate_score", -1))
            high_perf_filename = (
                f"{safe_candidate_name}_{new_id}_{new_candidate_score}分_胜出.txt"
            )
            dest_path = HIGH_PERFORMERS_DIR / high_perf_filename
            dest_path.write_text(resume_text, encoding="utf-8")
            print(
                f"   -> ✅ 简历已同时保存至: {dest_path.parent.name}/{dest_path.name}"
            )

    # --- 【【【 归档 3: 潜力候选人 (未参与对比，但分数 > 60) 】】】 ---
    elif not comparison_happened and final_score_value > 60:
        print(
            f"\n{Colors.YELLOW}>> [归档] 分数 ({final_score_value}%) > 60，正在保存为【潜力候选人】...{Colors.RESET}"
        )
        dest_filename = f"{safe_candidate_name}_{new_id}_{final_score_str}.txt"
        dest_path = POTENTIAL_CANDIDATES_DIR / dest_filename
        dest_path.write_text(resume_text, encoding="utf-8")
        print(f"   -> ✅ 简历已保存至: {dest_path.parent.name}/{dest_path.name}")

    elif ACTIVE_COMPARISON_TASK:
        print(
            f"{Colors.YELLOW}>> [Decision] Score ({final_score_value}%) < 70%. Skipping comparison analysis.{Colors.RESET}"
        )


# ===================================================================
# ===================================================================
# --- [MODIFIED] 完整替换此函数 ---
async def background_storage_task(
    new_id,
    candidate_name,
    phone,
    email,
    status,
    vector_score,
    final_match_score,
    core_strengths,
    core_gaps,
    ai_thinking_process,
    ai_analysis,
    raw_resume,
):
    """将所有提取和分析出的数据存入SQLite数据库。"""
    try:
        with sqlite3.connect(LOCAL_DATABASE_FILE) as conn:
            insert_query = """
            INSERT INTO reports (
                total_id, candidate_name, phone, email, status,
                vector_score, final_match_score, core_strengths, core_gaps,
                ai_thinking_process, ai_analysis, raw_resume, creation_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                new_id,
                candidate_name,
                phone,
                email,
                status,
                vector_score,
                final_match_score,
                core_strengths,
                core_gaps,
                ai_thinking_process,
                ai_analysis,
                raw_resume,
                datetime.now(timezone(timedelta(hours=8))).isoformat(),
            )
            conn.execute(insert_query, params)
            print(f" > [Sync] ID {new_id} ({candidate_name}) report saved to SQLite.")
    except Exception as e:
        print(f"!! [Sync Error] SQLite write failed: {e}")
        # 打印出参数，方便调试
        print(f" -> Failed params: id={new_id}, name={candidate_name}")


# ===================================================================================
# --- 后台任务与主循环 ---
# ===================================================================================


class AppController:
    def __init__(self):
        self.should_exit = False

    def trigger_exit(self):
        self.should_exit = True


class AsyncSafeCounter:
    def __init__(self, start_id=0):
        self.value = start_id
        self.lock = asyncio.Lock()

    async def get_next(self):
        async with self.lock:
            self.value += 1
            return self.value


class InboxHandler(FileSystemEventHandler):
    def __init__(self, loop, queue):
        self.loop = loop
        self.queue = queue
        self.last_proc = {}

    def on_created(self, event):
        if event.is_directory:
            return
        fp = Path(event.src_path)
        now = time.time()
        if fp in self.last_proc and now - self.last_proc[fp] < 1.0:
            return
        self.last_proc[fp] = now
        asyncio.run_coroutine_threadsafe(self.process_file(fp), self.loop)

    async def process_file(self, fp: Path):
        try:
            await asyncio.sleep(0.5)
            if not fp.exists():
                return
            print(f"\n>> [Inbox] Detected: {fp.name}")
            ext = fp.suffix.lower()
            if ext in [".txt", ".md", ".rtf"]:
                await self.queue.put(fp.read_text("utf-8"))
                shutil.move(str(fp), ARCHIVE_DIR / fp.name)
            elif ext in [".png", ".jpg", ".jpeg"]:
                await self.queue.put(fp)
            else:
                print(f" -> 🟡 Unsupported type '{ext}'.")
        except Exception as e:
            print(f"!! [File Error] {e}")


# ===================================================================================
# --- 【【【 在这里粘贴新的“批处理完成”监视器 】】】 ---
# ===================================================================================
async def queue_status_monitor(queue: asyncio.Queue):
    """
    一个后台监视器，当任务队列从“忙碌”变“空闲”时发送通知。
    """
    was_busy = False
    processed_count_in_batch = 0

    while True:
        await asyncio.sleep(2)  # 每2秒检查一次状态

        # _unfinished_tasks 是 queue.join() 内部使用的计数器，非常可靠
        is_currently_busy = queue._unfinished_tasks > 0

        if is_currently_busy:
            was_busy = True
            # 当队列忙碌时，我们可以通过队列大小来近似计算已处理数量
            # 这不是100%精确，但是一个很好的估算
            processed_count_in_batch = max(processed_count_in_batch, queue.qsize())

        # 核心逻辑：如果之前是忙的，但现在不忙了，说明一批任务处理完了
        elif not is_currently_busy and was_busy:
            # 等待一小会儿，确保所有任务的 "task_done()" 都已完全处理
            await asyncio.sleep(0.5)

            notification_title = "所有任务已处理完毕"
            notification_message = "简历处理队列现在为空，您可以继续添加新文件了。"

            # 使用 run_in_executor 在后台线程中安全地发送通知
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, safe_notification, notification_title, notification_message
            )

            # 重置状态，为下一批任务做准备
            was_busy = False
            processed_count_in_batch = 0


# --- 【【【 V3 MODIFIED: 将此函数改造为 async def 】】】 ---
async def consumer_worker_async(queue, counter):
    while not app_controller.should_exit:
        try:
            # 使用 await 来异步地从队列中获取任务
            content = await queue.get()

            # 异步地获取下一个ID
            new_id = await counter.get_next()

            # 异步地调用主分析函数
            # trigger_smart_analysis_async 内部现在包含了 await 调用，所以它必须被 await
            await trigger_smart_analysis_async(content, new_id)

            # 标记任务完成
            queue.task_done()

        except asyncio.CancelledError:
            # 捕获取消错误并正常退出循环
            break
        except Exception:
            # 捕获其他所有异常，打印堆栈信息以便调试
            print(f"!! [Worker Error]\n{traceback.format_exc()}")
            # 即使有错误，也要确保 task_done 被调用，避免队列卡死
            if "queue" in locals() and not queue.empty():
                queue.task_done()


# ===================================================================================
# --- 【【【 V2 新增：对比任务汇总报告生成器 】】】 ---
# ===================================================================================


# --- 新函数2：Qdrant 初始化函数 ---
# --- [MODIFIED V2] 完整替换此函数 ---
# ===================================================================================
# --- 【【【 Qdrant 连接最终解决方案：使用此函数替换旧版本 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 Qdrant 诊断版：用这个函数完整替换旧版本 】】】 ---
# ===================================================================================
def setup_qdrant():
    """初始化 Qdrant 客户端，并通过临时移除环境变量来确保不使用代理。"""
    global QDRANT_CLIENT
    if not QDRANT_AVAILABLE:
        return

    # --- 【【【 诊断探针 】】】 ---
    # 如果您在日志中看到了下面这行，就说明这个正确的函数被执行了！
    print(
        f"{Colors.MAGENTA}{Colors.BOLD}>>> 正在执行 Qdrant 连接诊断程序 v2...{Colors.RESET}"
    )

    # 在连接 Qdrant 之前，暂时从系统中移除代理设置
    original_proxies = {
        "HTTP_PROXY": os.environ.pop("HTTP_PROXY", None),
        "HTTPS_PROXY": os.environ.pop("HTTPS_PROXY", None),
        "ALL_PROXY": os.environ.pop("ALL_PROXY", None),
        "http_proxy": os.environ.pop("http_proxy", None),
        "https_proxy": os.environ.pop("https_proxy", None),
        "all_proxy": os.environ.pop("all_proxy", None),
    }

    # 打印出我们移除了什么，方便调试
    removed_keys = [k for k, v in original_proxies.items() if v is not None]
    if removed_keys:
        print(f"    -> [诊断] 临时移除了代理环境变量: {', '.join(removed_keys)}")
    else:
        print(f"    -> [诊断] 未发现需要移除的代理环境变量。")

    try:
        print(">> [Qdrant] Connecting to local Qdrant service (localhost:6333)...")

        # 现在调用 QdrantClient 时，它看不到任何代理设置，只能直连本地
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)

        collections = QDRANT_CLIENT.get_collections().collections
        collection_names = [c.name for c in collections]

        test_vector = EMBEDDING_MODEL.embed_query("test")
        vector_size = len(test_vector)

        if QDRANT_COLLECTION_NAME not in collection_names:
            print(
                f">> [Qdrant] Collection '{QDRANT_COLLECTION_NAME}' not found, creating..."
            )
            QDRANT_CLIENT.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

        if QDRANT_COMPARISON_COLLECTION_NAME not in collection_names:
            print(
                f">> [Qdrant] Collection '{QDRANT_COMPARISON_COLLECTION_NAME}' not found, creating..."
            )
            QDRANT_CLIENT.create_collection(
                collection_name=QDRANT_COMPARISON_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

        resume_count = QDRANT_CLIENT.count(
            collection_name=QDRANT_COLLECTION_NAME, exact=True
        ).count
        comparison_count = QDRANT_CLIENT.count(
            collection_name=QDRANT_COMPARISON_COLLECTION_NAME, exact=True
        ).count

        print(f"✅ [Qdrant] Connection successful.")
        print(
            f"   -> Resumes Collection: '{QDRANT_COLLECTION_NAME}' ({resume_count} vectors)"
        )
        print(
            f"   -> Comparisons Collection: '{QDRANT_COMPARISON_COLLECTION_NAME}' ({comparison_count} vectors)"
        )

    except Exception as e:
        print(f"❌ [Qdrant] FATAL ERROR: Could not connect to Qdrant. {e}")
        QDRANT_CLIENT = None
    finally:
        # 无论成功失败，都必须把代理环境变量恢复回去
        restored_keys = []
        for key, value in original_proxies.items():
            if value:
                os.environ[key] = value
                restored_keys.append(key)
        if restored_keys:
            print(f"    -> [诊断] 已恢复代理环境变量: {', '.join(restored_keys)}")


# --- 新函数3：简历向量化与存储函数 ---
async def vectorize_and_store_resume_async(candidate_id, name, resume_text):
    """将简历文本向量化并存入独立的 Qdrant 数据库。"""
    if not QDRANT_CLIENT or not resume_text:
        return

    try:
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(
            None, EMBEDDING_MODEL.embed_query, resume_text
        )
        point_id = str(uuid.uuid4())

        upsert_operation = lambda: QDRANT_CLIENT.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "candidate_id": candidate_id,
                        "candidate_name": name,
                        "text_snippet": resume_text[:500],
                    },
                )
            ],
            wait=True,
        )

        await loop.run_in_executor(None, upsert_operation)
        print(
            f" > [Qdrant Sync] ID {candidate_id} ({name}) resume vectorized and saved."
        )
    except Exception as e:
        print(f"!! [Qdrant Sync Error] Failed to vectorize for ID {candidate_id}: {e}")


# ===================================================================
# --- 【【【 复制到这里结束 】】】 ---
# ===================================================================
# --- 【【【 V3.6 FINAL: 唯一的、合并的最终报告生成器 】】】 ---
# --- 【【【 V4.0 FINAL: 替换此函数以生成信息更丰富的统一报告 】】】 ---
# --- 【【【 V4.1 FINAL: 替换此函数以生成信息最详尽的统一报告 】】】 ---
# --- 【【【 V4.2 FINAL: 替换此函数以实现报告“美颜”功能 】】】 ---
# ===================================================================
# --- 【【【 使用下面这个最终版本，替换您现有的同名函数 】】】 ---
# ===================================================================


# ===================================================================================
# --- 【【【 第四步：用这个全新的美化版报告函数，完整替换旧的 generate_combined_summary_report 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 使用下面这个最终版本，替换您现有的同名函数 】】】 ---
# ===================================================================================
# ===================================================================================
# --- 【【【 第3步：用这个最终修正版，完整替换您现有的同名函数 】】】 ---
# ===================================================================================
async def generate_combined_summary_report(
    analysis_results: list, comparison_results: list
):
    """
    在程序退出时，完成两项核心任务:
    1. 生成一个【智能动态】的统一汇总报告：对PK参与者展示包括简历在内的完整档案，对其他人只做分析摘要。
    2. 创建一个包含【所有候选人带排名原始简历】的独立存档文件夹（此功能不变）。
    """
    run_timestamp = datetime.now()
    timestamp_str = run_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    folder_timestamp_str = run_timestamp.strftime("%Y-%m-%d_%H%M%S")

    def safe_float(s):
        try:
            s_cleaned = re.sub(r"[^0-9.]", "", str(s))
            return float(s_cleaned)
        except (ValueError, TypeError):
            return -1.0

    # --- 1. 数据准备与全局排序 ---
    all_candidates_sorted = sorted(
        analysis_results, key=lambda x: safe_float(x.get("score")), reverse=True
    )

    # 将PK结果和完整的初步分析结果合并，方便后续调用
    analysis_map_by_name = {r.get("name"): r for r in analysis_results}
    pk_participants_full_data = []
    for task in comparison_results:
        new_candidate_name = task.get("new_candidate_name")
        if not new_candidate_name:
            continue
        full_analysis_data = analysis_map_by_name.get(new_candidate_name)
        if not full_analysis_data:
            continue
        pk_participants_full_data.append({**task, **full_analysis_data})

    # --- 任务1: 生成智能动态的统一汇总分析报告 ---
    report_lines = ["=" * 35 + " AI 招聘助理 - 统一运行汇总报告 " + "=" * 35]
    report_lines.append(f"报告生成时间: {timestamp_str}")
    report_lines.append("-" * 95)

    # Part 1: 全局统计概览
    report_lines.append("【本次运行全局概览】")
    report_lines.append(f"  - 总计分析候选人: {len(analysis_results)} 名")
    report_lines.append(
        f"  - 进入对比(PK)环节 (分数>=70%): {len(pk_participants_full_data)} 名"
    )
    report_lines.append("=" * 95)

    # Part 2: 所有候选人排名摘要 (此部分对所有人都不含简历，保持简洁)
    report_lines.append(
        f"\n\n--- 🚀 全局候选人排名与分析摘要 (共 {len(all_candidates_sorted)} 名) ---\n"
    )
    if not all_candidates_sorted:
        report_lines.append("本次运行未分析任何候选人。")
    else:
        for rank, candidate in enumerate(all_candidates_sorted, 1):
            rank_icon = {1: "👑 ", 2: "🥈 ", 3: "🥉 "}.get(rank, f"#{rank:<2}")
            report_lines.append(
                f"\n--- {rank_icon} | {candidate.get('name', 'N/A')} | 分数: {candidate.get('score', 'N/A')}% ---"
            )
            report_lines.append(
                f"  【核心匹配点】:\n{candidate.get('strengths', 'N/A').strip()}"
            )
            report_lines.append(
                f"  【核心风险点】:\n{candidate.get('gaps', 'N/A').strip()}"
            )
    report_lines.append("\n" + "=" * 95)

    # Part 3: 【【【 核心功能区 】】】 PK参与者完整档案 (作为附录，包含简历)
    report_lines.append(
        f"\n\n--- 附录: 重点候选人 (PK参与者) 完整档案 [{len(pk_participants_full_data)} 名] ---\n"
    )
    if not pk_participants_full_data:
        report_lines.append("本次运行无候选人进入PK环节。")
    else:
        sorted_pk_participants = sorted(
            pk_participants_full_data,
            key=lambda x: safe_float(x.get("score")),
            reverse=True,
        )
        for i, candidate in enumerate(sorted_pk_participants, 1):
            report_lines.append(
                f"\n"
                + "-" * 35
                + f" [重点档案 #{i}] {candidate.get('name', 'N/A')} "
                + "-" * 35
            )
            report_lines.append(f"\n【1. AI初步分析】")
            report_lines.append(f"  - 初步评分: {candidate.get('score', 'N/A')}%")
            report_lines.append(
                f"  - 完整分析:\n    核心匹配点:\n{candidate.get('strengths', 'N/A').strip()}\n\n    核心风险点:\n{candidate.get('gaps', 'N/A').strip()}"
            )

            report_lines.append(f"\n【2. AI对比分析】")
            report_lines.append(
                f"  - PK 评分: {candidate.get('new_candidate_score')} (基准: {candidate.get('benchmark_score')})"
            )
            report_lines.append(f"  - 最终裁决: {candidate.get('verdict', 'N/A')}")
            report_lines.append(
                f"  - 详细对比报告:\n{candidate.get('pk_report', 'N/A').strip()}"
            )

            report_lines.append(f"\n【3. 原始简历全文】")
            report_lines.append(candidate.get("raw_resume", "简历文本未能获取").strip())
            report_lines.append("-" * 95)

    # --- 保存统一汇总报告 ---
    full_report_str = "\n".join(report_lines)
    print("\n\n" + "=" * 35 + " 本次运行最终汇总 " + "=" * 35)
    print("\n".join(report_lines[:25] + ["\n...", "报告详情请查看文件。"]))
    print("=" * 95)
    try:
        SUMMARY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        main_report_path = (
            SUMMARY_REPORT_DIR / f"统一汇总报告_{folder_timestamp_str}.txt"
        )
        clean_report_str = re.sub(
            r"(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]", "", full_report_str
        ).replace("**", "")
        main_report_path.write_text(clean_report_str, encoding="utf-8")
        print(
            f"\n✅ {Colors.BOLD}{Colors.GREEN}统一汇总分析报告已成功保存。{Colors.RESET}"
        )
    except Exception as e:
        print(
            f"\n❌ {Colors.BOLD}{Colors.RED}保存统一汇总报告时出错: {e}{Colors.RESET}"
        )

    # --- 任务2: 创建【所有候选人】的带排名原始简历存档 (此功能保持不变，作为完整备份) ---
    if not all_candidates_sorted:
        return
    session_resume_dir = RANKED_RESUMES_DIR / f"{folder_timestamp_str}_本批次简历"
    try:
        session_resume_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ {Colors.BOLD}{Colors.RED}创建简历存档文件夹失败: {e}{Colors.RESET}")
        return

    print(
        f"\n>> [简历存档] 正在为 {len(all_candidates_sorted)} 位候选人生成带排名的原始简历文件..."
    )
    for rank, candidate_data in enumerate(all_candidates_sorted, 1):
        try:
            candidate_name = candidate_data.get("name", "未知姓名")
            score_str = str(candidate_data.get("score", "N_A")).replace("%", "")
            raw_resume_text = candidate_data.get("raw_resume", "")

            # 【【【 核心修正点：在这里统一调用净化函数 】】】
            safe_name = sanitize_filename(candidate_name)

            file_name = f"[Rank_{rank:02d}][{score_str}%][{safe_name}].txt"
            file_path = session_resume_dir / file_name
            file_path.write_text(raw_resume_text, encoding="utf-8")
        except Exception as e:
            print(f"   -> ❌ 保存候选人 '{candidate_name}' 的简历存档失败: {e}")
    print(f"✅ {Colors.BOLD}{Colors.GREEN}所有原始简历已按排名存档！{Colors.RESET}")
    print(
        f"📂 {Colors.BOLD}{Colors.CYAN}请查看文件夹: {session_resume_dir}{Colors.RESET}"
    )
    # --- 【【【 新增功能区结束 】】】 ---
    # --- 【【【 新增报告 2: 潜力候选人报告 (Potential Candidates) 】】】 ---
    # ... (这部分逻辑可以保持不变或按需修改) ...


# async def main():  <-- 确保你把上面的代码粘贴到这个函数的前面


# --- 【【【 替换结束 】】】 ---
# --- 【【【 V2 MODIFIED: 替换此函数以激活对比系统 】】】 ---
# --- 【【【 V3 MODIFIED: 替换此函数以激活新的“预加载”工作流 】】】 ---
# --- 【【【 V3.2 MODIFIED: 替换此函数以修正bug 】】】 ---
# --- 【【【 V3.3 MODIFIED: 替换此函数以使用简化的文件名前缀匹配 】】】 ---
# --- 【【【 V3.6 MODIFIED: 替换此函数以完成最终修正 】】】 ---
# --- 【【【 V3.6 FINAL: 最终毕业版的 main 函数 】】】 ---
# ===================================================================================
# --- 【【【 用这个完全修正后的版本，替换你现有的整个 main 函数 】】】 ---
# ===================================================================================
async def main():
    global app_controller, candidate_id_counter, ACTIVE_COMPARISON_TASK
    app_controller = AppController()

    # 在这里确保所有需要的文件夹都被创建
    for d in [
        IPC_DIR,
        INBOX_DIR,
        ARCHIVE_DIR,
        HIGH_PERFORMERS_DIR,
        SUMMARY_REPORT_DIR,
        COMPARISON_PARTICIPANTS_DIR,
        POTENTIAL_CANDIDATES_DIR,
        RANKED_RESUMES_DIR,  # <--- 【【【 把这一行加进去 】】】
    ]:
        d.mkdir(exist_ok=True)
    # ... (函数其余部分保持不变)

    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 10 + " AI Recruiter v33.1 - 【V4.0 智能决策版】 " + "=" * 10)
    print(">> System boot sequence initiated...")
    print("-" * 70)

    if not (
        setup_local_database() and setup_api_and_embedder() and load_and_vectorize_jds()
    ):
        input("Core system initialization failed. Press Enter to exit.")
        sys.exit(1)

    setup_qdrant()

    # --- 核心逻辑: 预加载对比任务模板 ---
    COMPARISON_DIR = Path(__file__).resolve().parent / "comparison_tasks"
    COMPARISON_DIR.mkdir(exist_ok=True)

    active_task_path = next(
        (
            d
            for d in COMPARISON_DIR.iterdir()
            if d.is_dir() and d.name != "completed_tasks"
        ),
        None,
    )

    if active_task_path:
        print(
            f"\n{Colors.MAGENTA}>> [对比系统] 检测到激活的对比任务: '{active_task_path.name}'{Colors.RESET}"
        )

        jd_file = next((f for f in active_task_path.glob("jd*.txt")), None)
        bench_file = next((f for f in active_task_path.glob("rs*.txt")), None)

        if jd_file and bench_file:
            jd_text = read_text_file_safely(jd_file)
            benchmark_text = read_text_file_safely(bench_file)
            if jd_text and benchmark_text:
                ACTIVE_COMPARISON_TASK = {
                    "task_name": active_task_path.name,
                    "jd_name": jd_file.name,
                    "jd_text": jd_text,
                    "benchmark_resume_text": benchmark_text,
                    "benchmark_name": parse_candidate_name(bench_file.name),
                }
                print(
                    f"{Colors.GREEN}   -> ✅ 模板加载成功！现在评级超过70分的简历都将与 '{bench_file.name}' 进行对比。{Colors.RESET}"
                )
            else:
                print(
                    f"{Colors.RED}   -> ❌ 错误：无法读取模板文件内容。对比功能将不被激活。{Colors.RESET}"
                )
        else:
            print(
                f"{Colors.YELLOW}   -> ⚠️ 警告：在任务文件夹中未找到必需的以 'jd' 和 'rs' 开头的文件。对比功能将不被激活。{Colors.RESET}"
            )
    else:
        print(
            f"\n{Colors.CYAN}>> [对比系统] 未发现激活的对比任务。程序将仅执行标准的单简历分析。{Colors.RESET}"
        )

    # --- 核心逻辑: 启动后台工作队列和文件监控 ---
    start_id = get_current_max_total_id()
    candidate_id_counter = AsyncSafeCounter(start_id=start_id)

    # 设置一个合理的并发数，避免压垮本地模型
    num_workers = 1
    print(f">> [System] Starting up with {num_workers} concurrent workers...")
    tasks = [
        asyncio.create_task(consumer_worker_async(task_queue, candidate_id_counter))
        for _ in range(num_workers)
    ]
    tasks.append(asyncio.create_task(queue_status_monitor(task_queue)))

    handler = InboxHandler(asyncio.get_running_loop(), task_queue)
    observer = None
    if WATCHDOG_AVAILABLE:
        observer = Observer()
        observer.schedule(handler, str(INBOX_DIR), recursive=False)
        observer.start()
    else:
        print(
            f"{Colors.YELLOW}>> [Warning] Watchdog not installed. File monitoring is disabled.{Colors.RESET}"
        )

    print(f"\n>> [System Ready] Monitoring inbox: {INBOX_DIR}")
    print("\n  Press Ctrl+C to safely shut down.")
    print("=" * 70)

    # --- 核心逻辑: 主循环与优雅退出处理 ---
    try:
        while not app_controller.should_exit:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        app_controller.trigger_exit()
    finally:
        print("\n>> Shutting down AI Recruiter...")
        if observer:
            observer.stop()
            observer.join()

        print(">> Waiting for all pending tasks to complete...")
        await task_queue.join()

        print(">> Cancelling worker tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # --- 调用唯一的、统一的报告生成器 ---
        await generate_combined_summary_report(
            all_session_results, all_comparison_results
        )

        print("\n✅ AI Recruiter has been safely shut down.")


# ===================================================================================
# --- 【【【 V-API 新增：FastAPI 服务器 & Tampermonkey 接口 】】】 ---
# ===================================================================================


# 1. 定义 API 的响应数据模型 (使用 Pydantic)
#    这能确保我们的 API 返回的数据格式永远是标准的
class CandidateInfoResponse(BaseModel):
    name: str = Field(..., description="候选人姓名")
    score: float = Field(..., description="AI给出的最终匹配分数")
    best_position: str = Field(..., description="AI认为最匹配的职位")
    recommendation: str = Field(..., description="AI的最终推荐建议")
    market_competitiveness: str = Field(..., description="AI对候选人市场竞争力的评估")


# 2. 创建 FastAPI 应用实例
app = FastAPI(
    title="AI 招聘助理 API (v33.1 增强版)",
    description="为浏览器插件提供候选人分析数据的后端服务",
    version="1.0.0",
)

# 3. 添加 CORS 中间件，允许所有来源的跨域请求 (让插件可以访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. 核心功能函数：根据姓名在已分析的结果中查找数据
def find_candidate_analysis(name_from_web: str):
    """
    在内存中的 all_session_results 列表中查找匹配的候选人数据。
    """
    if not name_from_web:
        return None

    normalized_web_name = re.sub(
        r"[\s\W_]+", "", unicodedata.normalize("NFKC", name_from_web)
    ).lower()
    if not normalized_web_name:
        return None

    # 从最新的结果开始往前找，以获取最新的分析
    for result in reversed(all_session_results):
        name_from_db = result.get("name")
        if name_from_db:
            normalized_db_name = re.sub(
                r"[\s\W_]+", "", unicodedata.normalize("NFKC", name_from_db)
            ).lower()

            # 模糊匹配：只要一个名字包含另一个即可
            if (
                normalized_web_name in normalized_db_name
                or normalized_db_name in normalized_web_name
            ):

                # 从复杂的分析结果中，提取出插件需要的几个关键字段
                score_text = result.get("score", "0")
                try:
                    score = float(re.sub(r"[^0-9.]", "", str(score_text)))
                except:
                    score = 0.0

                # 尝试从 PK 结果中获取更精确的推荐，如果没有则从单人分析中获取
                final_recommendation = "N/A"
                if all_comparison_results:
                    for pk_result in reversed(all_comparison_results):
                        if pk_result.get("new_candidate_name") == name_from_db:
                            final_recommendation = pk_result.get(
                                "pk_report", "分析报告不完整"
                            )
                            break  # 找到最新的PK报告就停止

                # 如果没有PK报告，就用核心优劣势作为推荐
                if final_recommendation == "N/A":
                    strengths = result.get("strengths", "无").strip()
                    gaps = result.get("gaps", "无").strip()
                    final_recommendation = (
                        f"【核心匹配点】:\n{strengths}\n\n【核心风险点】:\n{gaps}"
                    )

                # 暂时将 market_competitiveness 设为 N/A，因为当前 prompt 不直接输出这个字段
                # 如果需要，未来可以修改 prompt 来提取
                return {
                    "name": name_from_db,
                    "score": score,
                    "best_position": next(
                        iter(result.get("vector_scores", {"N/A": 0}))
                    ),  # 获取向量匹配度最高的职位
                    "recommendation": final_recommendation,
                    "market_competitiveness": "N/A",
                }
    return None


# 5. 定义 API 端点 (Endpoint)
@app.get("/get_candidate_info", response_model=CandidateInfoResponse)
async def get_candidate_info(name: str):
    """
    这是 Tampermonkey 插件调用的接口。
    它接收一个姓名，然后返回该候选人的 AI 分析结果。
    """
    analysis_data = find_candidate_analysis(name)
    if analysis_data:
        return analysis_data
    else:
        # 如果找不到，返回标准的 404 Not Found 错误
        raise HTTPException(
            status_code=404, detail=f"未在已分析的缓存中找到名为 '{name}' 的候选人数据"
        )


# 6. 定义一个函数来运行 API 服务器
def run_api_server():
    """
    在一个独立的线程中启动 FastAPI 服务器。
    """
    print("\n" + "=" * 70)
    print(f"  [FastAPI 服务器] {Colors.BOLD}{Colors.GREEN}启动成功！{Colors.RESET}")
    print(
        f"  - 插件接口正在监听: {Colors.CYAN}http://127.0.0.1:5003/get_candidate_info{Colors.RESET}"
    )
    print(
        f"  - API 交互文档请访问: {Colors.CYAN}http://127.0.0.1:5003/docs{Colors.RESET}"
    )
    print("=" * 70)
    # 使用 uvicorn 启动服务器，并指定我们想要的端口 5003
    uvicorn.run(app, host="127.0.0.1", port=5003, log_level="warning")


# --- 【【【 V-API 新增结束 】】】 ---
# --- 【【【 V3.5 MODIFIED: 替换此启动器以实现优雅退出 】】】 ---
# ===================================================================================
# --- 【【【 V-API 最终启动器 】】】 ---
# ===================================================================================
if __name__ == "__main__":
    # 1. 创建一个独立的后台线程来运行 API 服务器
    #    daemon=True 意味着当主程序退出时，这个线程也会自动结束
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    # 2. 在主线程中运行我们原来的异步主程序
    loop = asyncio.get_event_loop()
    main_task = loop.create_task(main())

    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print(
            "\n>> [System] KeyboardInterrupt detected. Initiating graceful shutdown..."
        )
        if "app_controller" in globals() and app_controller:
            app_controller.trigger_exit()
        try:
            loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            pass
    finally:
        print("\n>> [System] Finalizing shutdown sequence...")
        if loop.is_running():
            loop.close()
            print(">> [System] Event loop closed.")
        print(">> [System] Program has terminated gracefully.")
