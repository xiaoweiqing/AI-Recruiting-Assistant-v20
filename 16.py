#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===================================================================================
#      AI 招聘助理 v6.0 (基于 v31.0 重构终版)
# ===================================================================================
# 版本说明:
# - [架构回归] 以 v31.0 的稳定架构为基础，彻底解决模型加载死锁问题。
# - [功能嫁接] 完整移植 v4.x 的深度分析Prompt、JSON解析、三库同步和批量处理模式。
# - [稳定至上] 保留 v31.0 的初始化顺序和 asyncio 结构，确保启动流畅。
# - [体验完整] 这是集稳定性和强大功能于一身的最终版本。
# ===================================================================================
# --- 核心 import (源自 v31.0 并补充) ---
import asyncio
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import traceback
import unicodedata
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

# --- 代理清理 ---
for proxy_var in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

# --- AI 与数据相关库 ---
import notion_client
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient, models

try:
    import fitz
    import docx
    FILE_READERS_AVAILABLE = True
except ImportError:
    FILE_READERS_AVAILABLE = False

# ==============================================================================
# ⬇⬇⬇ 0. 用户配置区 (与 v4.x/v5.0 相同) ⬇⬇⬇
# ==============================================================================
load_dotenv()
LOCAL_API_BASE = "http://127.0.0.1:8087/v1"
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
ANALYSIS_DB_ID = os.getenv("CANDIDATE_DB_ID")
PROFILE_DB_ID = os.getenv("CANDIDATE_PROFILE_HUB_DB_ID")
BASE_DIR = Path(__file__).resolve().parent
JDS_FOLDER_PATH = BASE_DIR / "JDs_library"
RESUMES_DIR = BASE_DIR / "resumes_to_process"
PROCESSED_DIR = BASE_DIR / "processed_resumes"
FAILED_DIR = BASE_DIR / "processed_failed"
SQLITE_DB_FILE = BASE_DIR / "recruitment_data_v6.0.db"
QDRANT_COLLECTION_NAME = "ai_recruitment_assistant_v6_0"
WORKER_COUNT = 4


class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'


# ==============================================================================
# ⬇⬇⬇ 1. AI 指令 (Prompt - 移植自 v4.x) ⬇⬇⬇
# ==============================================================================
AI_PROMPT_TEMPLATE = """### INSTRUCTION:
You are a top-tier AI Technical Recruiter. Your task is to analyze the candidate's resume against the provided job descriptions. You must provide two distinct outputs: a structured JSON block for data extraction, followed by a detailed, free-form analysis in Markdown format.

**1. Structured Data Extraction (JSON Block):**
First, extract the key information from the resume. This block MUST start with ```json and end with ```. If a field is not found, use "N/A".

```json
{{
  "candidate_name": "Full name",
  "phone": "Phone number",
  "email": "Email address",
  "education": "Highest degree obtained",
  "work_experience_summary": "e.g., '5 years of backend development'",
  "current_location": "Current city",
  "expected_salary": "Expected salary if mentioned",
  "core_skills": "Comma-separated list of key technical skills",
  "final_match_score_percent": "A number from 0 to 100"
}}
2. In-depth Analysis Report (Markdown):
Next, provide a comprehensive analysis report in Markdown format. This should be your detailed thought process.
AI Analysis Report
Candidate Profile:
Name: {candidate_name}
Contact: {phone} | {email}
Core Competencies: {core_skills}
Positional Fit Analysis:
Best Matched Position: (Select the single best matching position from the Job Descriptions list)
Dimensional Analysis:
Educational Background: (Your detailed analysis on this dimension.)
Technical Skills & Experience: (Your detailed analysis on this dimension. Compare resume skills against JD requirements.)
Project Experience & Achievements: (Your detailed analysis on this dimension.)
Potential Gaps / Risks: (Highlight 1-2 potential weaknesses or areas for further questioning.)
Final Verdict:
Overall Summary: (A concluding paragraph summarizing the candidate's overall fit and potential.)
Recommendation: (A single, actionable sentence, e.g., "Highly recommend for interview.")
Final Match Score: {final_match_score_percent}%
JOB DESCRIPTIONS:
{jd_input}
CANDIDATE RESUME:
{resume_text}
YOUR FULL RESPONSE (JSON, then Markdown Report):
"""
NOTION_PROPS = {
    "analysis_name": "候选人姓名", "analysis_date": "分析日期", "analysis_source": "源文件名",
    "analysis_score": "综合匹配分", "profile_name": "姓名", "profile_relation_to_analysis": "关联分析报告",
    "profile_phone": "联系电话", "profile_email": "候选人邮箱", "profile_core_skills": "核心能力或技能",
    "profile_experience": "岗位经验", "profile_education": "学历", "profile_current_location": "目前所在地",
    "profile_expected_salary": "期望薪资",
}

# 全局变量
llm, EMBEDDING_MODEL, QDRANT_CLIENT, ACTIVE_JD_DATA = None, None, None, {}
IS_RAG_ENABLED = False


# ==============================================================================
# ⬇⬇⬇ 2. 初始化与设置模块 (源自 v31.0 的稳定结构) ⬇⬇⬇
# ==============================================================================
def setup_sqlite():
    try:
        print(">> [SQLite] 正在初始化本地数据库...")
        with sqlite3.connect(SQLITE_DB_FILE) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY, candidate_name TEXT, match_score TEXT,
                notion_analysis_page_id TEXT, notion_profile_page_id TEXT,
                ai_full_report TEXT, extracted_json TEXT, resume_text TEXT,
                source_file TEXT, creation_time TEXT
            )""")
        print(f"✅ [SQLite] 数据库 '{SQLITE_DB_FILE.name}' 准备就绪。")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ [Fatal] SQLite 初始化失败: {e}{Colors.RESET}")
        return False


def setup_api_and_embedder():
    """
    【【【 核心函数 - 源自 v31.0 】】】
    严格保持这个被验证过可以成功运行的初始化顺序。
    """
    global llm, EMBEDDING_MODEL
    try:
        # 1. 先初始化 API
        print(f">> [AI Engine] 正在连接本地 API: {LOCAL_API_BASE}...")
        llm = ChatOpenAI(openai_api_base=LOCAL_API_BASE, openai_api_key="na", model_name="local", temperature=0.1, max_tokens=-1, request_timeout=600)
        llm.invoke("Hi")
        print("✅ [AI Engine] API 连接成功。")

        # 2. 再初始化 Embedding
        print(f">> [Embedding] 正在加载 HuggingFace 嵌入模型: {EMBEDDING_MODEL_NAME}...")
        EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
        print("✅ [Embedding] 嵌入模型加载成功。")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ [Fatal] AI 引擎或嵌入模型初始化失败: {e}{Colors.RESET}")
        return False


def setup_qdrant():
    global QDRANT_CLIENT, IS_RAG_ENABLED
    try:
        print(">> [Qdrant] 正在连接向量数据库...")
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
        QDRANT_CLIENT.get_collections()
        IS_RAG_ENABLED = True
        print("✅ [Qdrant] 向量数据库连接成功。")
        return True
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️ [Warning] Qdrant 连接失败: {e} (RAG功能将禁用){Colors.RESET}")
        return True


def load_jds() -> bool:
    global ACTIVE_JD_DATA
    print(f">> [JD Loader] 正在加载并向量化 JD...")
    JDS_FOLDER_PATH.mkdir(exist_ok=True)
    jds = {}
    for fp in JDS_FOLDER_PATH.glob("*.txt"):
        try:
            content = fp.read_text('utf-8')
            # 使用同步方法
            jds[fp.stem] = {"content": content, "vector": EMBEDDING_MODEL.embed_query(content)}
        except Exception as e:
            print(f" ❌ 加载 '{fp.name}' 失败: {e}")
    ACTIVE_JD_DATA = jds
    print(f"✅ [JD Loader] 加载并向量化了 {len(jds)} 份 JD。")
    return bool(jds)


# ==============================================================================
# ⬇⬇⬇ 3. 核心分析与数据处理模块 (移植自 v4.x) ⬇⬇⬇
# ==============================================================================
async def process_resume(file_path: Path, worker_name: str):
    filename = file_path.name
    print(f"\n>> [{worker_name}] 开始处理简历: {Colors.BOLD}{filename}{Colors.RESET}")
    try:
        resume_text = await asyncio.to_thread(read_file_content, file_path)
        if not resume_text or len(resume_text.strip()) < 50:
            return "skipped_empty"

        jd_input = "\n\n---\n\n".join([f"## {title}\n{data['content']}" for title, data in ACTIVE_JD_DATA.items()])
        if not jd_input:
            return "skipped_no_jd"

        prompt = AI_PROMPT_TEMPLATE.format(jd_input=jd_input, resume_text=resume_text[:8000])

        print(f"   -> [AI] 正在请求深度分析 (流式)...")
        full_response = ""
        try:
            async for chunk in llm.astream(prompt):
                print(chunk.content, end='', flush=True)
                full_response += chunk.content
        except Exception as e:
            print(f"\n{Colors.RED}!! [AI Error] 模型调用失败: {e}{Colors.RESET}")
            raise

        print()
        json_data, markdown_report = parse_ai_response(full_response)
        if not json_data or not markdown_report:
            raise ValueError("AI response parsing failed")

        name, score = json_data.get('candidate_name', '未知'), json_data.get('final_match_score_percent', 0)
        print(f"   -> {Colors.GREEN}✅ [Done] 分析完成: {name} | 匹配分: {score}%{Colors.RESET}")
        
        # 后台存储任务
        notion_page_ids = await save_to_notion_async(json_data, markdown_report, filename)
        await save_to_sqlite_async(json_data, markdown_report, resume_text, filename, notion_page_ids)
        if notion_page_ids and notion_page_ids[1]:
            asyncio.create_task(vectorize_to_qdrant_async(notion_page_ids[1], json_data))
        return {"status": "success", "name": name, "score": score}
    except Exception as e:
        print(f"{Colors.RED}!! [处理失败] 简历 '{filename}' 处理时发生错误: {e}{Colors.RESET}")
        return {"status": "error", "file": file_path, "traceback": traceback.format_exc()}


# ... (数据存储、文件读取、解析等所有辅助函数移植自 v4.x)
def parse_ai_response(full_response: str) -> (dict, str):
    json_data, markdown_report = {}, ""
    try:
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", full_response, re.DOTALL)
        if json_match:
            json_data = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        pass
    report_match = re.search(r"---[\s\n]*### AI Analysis Report", full_response, re.DOTALL)
    if report_match:
        markdown_report = full_response[report_match.start():]
    return json_data, markdown_report


def read_file_content(file_path: Path) -> str:
    try:
        ext = file_path.suffix.lower()
        if ext == '.pdf' and FILE_READERS_AVAILABLE:
            return "".join(page.get_text() for page in fitz.open(file_path))
        if ext == '.docx' and FILE_READERS_AVAILABLE:
            return "\n".join([p.text for p in docx.Document(file_path).paragraphs])
        if ext in ['.txt', '.md']:
            return file_path.read_text('utf-8', errors='ignore')
    except Exception as e:
        print(f"!! [File Reader] 读取 '{file_path.name}' 失败: {e}")
    return ""


async def save_to_notion_async(data, report_text, filename):
    try:
        async_notion = notion_client.AsyncClient(auth=NOTION_TOKEN)
        beijing_time = datetime.now(timezone(timedelta(hours=8)))
        name = data.get('candidate_name', '未知')
        analysis_props = {
            NOTION_PROPS["analysis_name"]: {"title": [{"text": {"content": f"{name} - AI分析报告"}}]},
            NOTION_PROPS["analysis_date"]: {"date": {"start": beijing_time.isoformat()}},
            NOTION_PROPS["analysis_source"]: {"rich_text": [{"text": {"content": filename}}]},
            NOTION_PROPS["analysis_score"]: {"number": int(data.get('final_match_score_percent', 0))}
        }
        report_blocks = [{"type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}} for chunk in [report_text[i:i + 1900] for i in range(0, len(report_text), 1900)]]
        new_analysis_page = await async_notion.pages.create(parent={"database_id": ANALYSIS_DB_ID}, properties=analysis_props, children=report_blocks)
        analysis_page_id = new_analysis_page.get('id')
        print(" -> [Sync] 已同步到 Notion。")
        
        def create_rich_text(value):
            return {"rich_text": [{"text": {"content": str(value)}}]} if value and value != "N/A" else None
        profile_props = {
            NOTION_PROPS["profile_name"]: {"title": [{"text": {"content": name}}]},
            NOTION_PROPS["profile_relation_to_analysis"]: {"relation": [{"id": analysis_page_id}]},
            NOTION_PROPS["profile_phone"]: {"phone_number": data.get('phone') if data.get('phone') != "N/A" else None},
            NOTION_PROPS["profile_email"]: {"email": data.get('email') if data.get('email') != "N/A" else None},
            NOTION_PROPS["profile_core_skills"]: create_rich_text(data.get("core_skills")),
            NOTION_PROPS["profile_education"]: create_rich_text(data.get("education")),
            NOTION_PROPS["profile_current_location"]: create_rich_text(data.get("current_location")),
            NOTION_PROPS["profile_expected_salary"]: create_rich_text(data.get("expected_salary")),
            NOTION_PROPS["profile_experience"]: create_rich_text(data.get("work_experience_summary")),
        }
        new_profile_page = await async_notion.pages.create(parent={"database_id": PROFILE_DB_ID}, properties={k: v for k, v in profile_props.items() if v})
        profile_page_id = new_profile_page.get('id')
        print(" -> [Sync] 已同步到 Notion。")
        return analysis_page_id, profile_page_id
    except Exception as e:
        print(f"{Colors.RED}!! [Sync Error] Notion 同步失败: {e}{Colors.RESET}")
        return None, None


async def save_to_sqlite_async(data, report_text, resume_text, filename, notion_ids):
    try:
        analysis_id, profile_id = notion_ids if notion_ids else (None, None)
        await asyncio.to_thread(_sqlite_write_op, data, report_text, resume_text, filename, analysis_id, profile_id)
        print(" -> [Sync] 已保存到 SQLite。")
    except Exception as e:
        print(f"{Colors.RED}!! [Sync Error] SQLite 写入失败: {e}{Colors.RESET}")


def _sqlite_write_op(data, report_text, resume_text, filename, analysis_id, profile_id):
    with sqlite3.connect(SQLITE_DB_FILE) as conn:
        conn.execute("""
        INSERT INTO reports (candidate_name, match_score, notion_analysis_page_id, notion_profile_page_id, ai_full_report, extracted_json, resume_text, source_file, creation_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (data.get('candidate_name'), f"{data.get('final_match_score_percent')}%", analysis_id, profile_id, report_text, json.dumps(data, ensure_ascii=False), resume_text, filename, datetime.now(timezone(timedelta(hours=8))).isoformat()))


async def vectorize_to_qdrant_async(profile_page_id, data):
    if not IS_RAG_ENABLED:
        return
    try:
        name = data.get('candidate_name', 'N/A')
        text_to_embed = f"Candidate: {name}\nCore Skills: {data.get('core_skills', 'N/A')}\nExperience: {data.get('work_experience_summary', 'N/A')}"
        vector = await asyncio.to_thread(EMBEDDING_MODEL.embed_query, text_to_embed)
        payload = {"candidate_name": name, "summary_text": text_to_embed}
        await asyncio.to_thread(QDRANT_CLIENT.upsert, QDRANT_COLLECTION_NAME, [models.PointStruct(id=profile_page_id, vector=vector, payload=payload)], wait=True)
        print(" -> [Sync] 已向量化到 Qdrant。")
    except Exception as e:
        print(f"{Colors.RED}!! [Sync Error] Qdrant 向量化失败: {e}{Colors.RESET}")


# ==============================================================================
# ⬇⬇⬇ 5. 任务调度与主循环模块 (移植自 v4.x 的并发模型) ⬇⬇⬇
# ==============================================================================
async def worker(name, queue, results):
    while True:
        try:
            file_path = await queue.get()
            try:
                result = await process_resume(file_path, name)
                results.append({"file": file_path, "result": result})
            except Exception as e:
                results.append({"file": file_path, "result": f"error_{traceback.format_exc()}"})
            finally:
                queue.task_done()
        except asyncio.CancelledError:
            break


async def run_batch_mode():
    print("\n" + "=" * 25 + " 批量简历处理模式 " + "=" * 25)
    for d in [RESUMES_DIR, PROCESSED_DIR, FAILED_DIR]:
        d.mkdir(exist_ok=True)
    files_to_process = [p for p in RESUMES_DIR.iterdir() if p.is_file()]
    if not files_to_process:
        print(f"\n>> 在 '{RESUMES_DIR.name}' 文件夹中没有简历。")
        return
    
    print(f"\n>> 发现 {len(files_to_process)} 份简历，启动 {WORKER_COUNT} 个并发工作单元...")
    queue = asyncio.Queue()
    results = []
    for f in files_to_process:
        await queue.put(f)

    worker_tasks = [asyncio.create_task(worker(f"Worker-{i+1}", queue, results)) for i in range(WORKER_COUNT)]
    await queue.join()
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)

    print("\n" + "=" * 30 + " 处理完成 " + "=" * 30)
    success, fail, skip = 0, 0, 0
    for item in results:
        file, result = item['file'], item['result']
        if isinstance(result, dict) and result.get("status") == "success":
            success += 1
            shutil.move(str(file), str(PROCESSED_DIR / file.name))
        elif isinstance(result, str) and result.startswith("skipped"):
            skip += 1
            shutil.move(str(file), str(PROCESSED_DIR / file.name))
        else:
            fail += 1
            shutil.move(str(file), str(FAILED_DIR / file.name))
    print(f"✅ 成功分析: {success} 份\n🟡 跳过处理: {skip} 份\n❌ 处理失败: {fail} 份")
    print(f"\n>> 已处理的文件已移动到 '{PROCESSED_DIR.name}' 和 '{FAILED_DIR.name}'。")


async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 12 + " AI 招聘助理 v6.0 (基于v31重构终版) " + "=" * 12)
    
    # 【【【 使用 v31.0 的串行初始化流程 】】】
    print(">> [System] 正在初始化所有服务...")
    if not (setup_sqlite() and setup_api_and_embedder() and setup_qdrant() and load_jds()):
        await asyncio.to_thread(input, "\n系统初始化失败，请检查配置。按回车键退出。")
        return
    
    print("✅ [System] 所有服务初始化成功！")
    print("-" * 70)
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 32 + " 主菜单 " + "=" * 33)
        print("\n  [ 1 ]  批量处理 'resumes_to_process' 文件夹")
        print("\n  [ Q ]  退出程序")
        print("\n" + "=" * 75)
        choice = await asyncio.to_thread(input, "请输入您的选择并按回车: ")
        
        if choice == '1':
            await run_batch_mode()
            await asyncio.to_thread(input, "\n>> 批量处理完成。按回车返回主菜单...")
        elif choice.lower() == 'q':
            break
        else:
            print("\n!! 无效输入，请重新选择。")
            await asyncio.sleep(2)

    print("\n>> [System] 感谢使用，正在安全退出...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n>> [System] 用户中断，强制退出。")
    finally:
        print(">> [System] 程序已关闭。")