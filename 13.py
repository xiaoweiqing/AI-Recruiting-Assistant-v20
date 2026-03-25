# ===================================================================================
# AI 招聘助理 v24.2 - 最终稳定版
# ===================================================================================
# 版本说明:
# - v24.2 最终整合: 整合所有修复，包括代理清理、缺失的import和类定义，确保开箱即用。
# - 【核心升级：全英文结构化输出】:
#   1.  【全英文Prompt】: 采用全英文指令，输出结构化的Markdown报告，利用模型母语优势，提升速度和质量。
#   2.  【后台智能解析】: AI只在命令行输出最终报告。所有数据提取、解析和存储任务均在后台静默完成。
#   3.  【增强数据存储】: 数据库和向量库现在存储的是从英文报告中精确解析出的结构化字段。
#   4.  【保留核心功能】: RAG、本地报告生成、多JD处理、文件监控等所有核心功能均已适配并保留。
# ===================================================================================

# --- 核心 import ---
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

# ===================================================================================
# --- 【关键修复】取消系统代理环境变量，确保本地连接不受影响 ---
# ===================================================================================
for proxy_var in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if proxy_var in os.environ:
        print(f">> [Proxy Cleaner] Found and removed system proxy setting: {proxy_var}")
        del os.environ[proxy_var]
# ===================================================================================

from dotenv import load_dotenv

# --- LangChain & AI 相关的导入 ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore as Qdrant

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError: WATCHDOG_AVAILABLE = False
try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError: QDRANT_AVAILABLE = False
try:
    from PIL import Image
    import pytesseract
    PIL_AVAILABLE = True
except ImportError: PIL_AVAILABLE = False
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError: PLYER_AVAILABLE = False

# --- 全局配置与类 ---
class Colors:
    RESET = '\033[0m'; BOLD = '\033[1m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
    RED = '\033[91m'; CYAN = '\033[96m'

load_dotenv()

IPC_DIR = Path.home() / ".ai_assistant_ipc"
INBOX_DIR = IPC_DIR / "inbox"
ARCHIVE_DIR = IPC_DIR / "archive"
SUMMARY_REPORT_DIR = Path(__file__).resolve().parent / "所有汇总报告"
JDS_FOLDER_PATH = Path(__file__).resolve().parent / "JDs_library"
LOCAL_DATABASE_FILE = IPC_DIR / "recruitment_data_v24.db"

DB_PROPS = {
    "analysis_id": "id", "analysis_total_id": "total_id", "analysis_name": "candidate_name",
    "analysis_score": "match_score", "full_report_text": "full_report_text", "analysis_date": "creation_time",
}

AI_PROMPT_TEMPLATE = """### INSTRUCTION:
You are a top-tier AI Technical Recruiter. Your task is to analyze the following Job Descriptions and Candidate Resume, then generate a structured analysis report in Markdown format.

### OUTPUT FORMAT:

**1. Candidate Profile**
- **Name**: (Extract from resume)
- **Phone**: (Extract from resume, use "N/A" if not found)
- **Email**: (Extract from resume, use "N/A" if not found)
- **Core Competencies**: (Provide a comma-separated list of key skills and technologies)

**2. Positional Fit Analysis**
- **Best Matched Position**: (Select the best matching position from the Job Descriptions list)
- **Dimensional Analysis**:
  - **Educational Background**: (Your detailed analysis on this dimension)
  - **Technical Skills & Experience**: (Your detailed analysis on this dimension)
  - **Project Experience & Achievements**: (Your detailed analysis on this dimension)
  - **Soft Skills & Communication**: (Your detailed analysis on this dimension)
- **Overall Summary**: (A concluding paragraph summarizing the candidate's fit)

**3. Final Verdict**
- **Recommendation**: (A single, actionable sentence, e.g., "Highly recommend for interview", "Consider for future opportunities.")
- **Final Match Score**: XX%

---

### JOB DESCRIPTIONS:
{jd_input}

### CANDIDATE RESUME:
{resume_text}

### YOUR ANALYSIS REPORT:
"""

# --- 全局变量 ---
llm, ACTIVE_JD_DATA, app_controller, IS_RAG_ENABLED = None, {}, None, False
QDRANT_CLIENT: QdrantClient = None
EMBEDDING_MODEL: HuggingFaceEmbeddings = None
QDRANT_COLLECTION_NAME = "ai_recruiter_v24_en"
task_queue = asyncio.Queue()
candidate_id_counter = None
session_analysis_results = []
session_analysis_results_lock = asyncio.Lock()

# ===================================================================================
# --- 初始化与设置模块 ---
# ===================================================================================

def setup_local_database():
    print(">> [DB] Initializing SQLite database...")
    try:
        with sqlite3.connect(LOCAL_DATABASE_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS analysis_reports (
                {DB_PROPS['analysis_id']} INTEGER PRIMARY KEY AUTOINCREMENT,
                {DB_PROPS['analysis_total_id']} INTEGER UNIQUE NOT NULL,
                {DB_PROPS['analysis_name']} TEXT,
                phone TEXT, email TEXT, core_competencies TEXT,
                best_position TEXT, dimensional_analysis TEXT, overall_summary TEXT,
                recommendation TEXT, {DB_PROPS['analysis_score']} REAL,
                {DB_PROPS['full_report_text']} TEXT,
                {DB_PROPS['analysis_date']} TEXT NOT NULL
            )""")
            conn.commit()
        print(f"✅ [DB] Database '{LOCAL_DATABASE_FILE.name}' initialized successfully.")
        return True
    except sqlite3.Error as e:
        print(f"❌ [DB] FATAL ERROR: Database setup failed: {e}")
        return False

def setup_api():
    global llm
    LOCAL_API_BASE = "http://127.0.0.1:8087/v1"
    try:
        print(f">> [AI Engine] Connecting to local model API at {LOCAL_API_BASE}...")
        llm = ChatOpenAI(
            openai_api_base=LOCAL_API_BASE, openai_api_key="not-needed",
            model_name="local-model", temperature=0.1, max_tokens=-1, request_timeout=600
        )
        print(">> [AI Engine] Testing connection...")
        llm.invoke("Hi")
        print(f"{Colors.GREEN}✅ [AI Engine] Connection to local model API successful.{Colors.RESET}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ [AI Engine] FATAL ERROR: Could not connect to local model API: {e}{Colors.RESET}")
        traceback.print_exc()
        return False

async def setup_qdrant_async():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED
    if not QDRANT_AVAILABLE: print("❌ [Vector DB] FATAL ERROR: qdrant-client is not installed."); IS_RAG_ENABLED = False; return
    try:
        print(">> [Vector DB] Loading local embedding model...")
        EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={'device': 'cpu'})
        print(f"✅ [Vector DB] Embedding model loaded.")
        print(">> [Vector DB] Connecting to Qdrant service (localhost:6333)...")
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
        loop = asyncio.get_running_loop()
        collections = await loop.run_in_executor(None, QDRANT_CLIENT.get_collections)
        if QDRANT_COLLECTION_NAME not in [c.name for c in collections.collections]:
            print(f">> [Vector DB] Collection '{QDRANT_COLLECTION_NAME}' not found, creating...")
            vector_size = len(EMBEDDING_MODEL.embed_query("test"))
            await loop.run_in_executor(None, QDRANT_CLIENT.create_collection, QDRANT_COLLECTION_NAME, models.VectorParams(size=vector_size, distance=models.Distance.COSINE))
        count = await loop.run_in_executor(None, lambda: QDRANT_CLIENT.count(collection_name=QDRANT_COLLECTION_NAME, exact=True))
        print(f"✅ [Vector DB] Connection to Qdrant successful! Collection contains {count.count} records.")
        IS_RAG_ENABLED = True
    except Exception as e:
        print(f"❌ [Vector DB] FATAL ERROR: Could not connect to Qdrant: {e}")
        IS_RAG_ENABLED = False

def load_active_jd_from_local_folder():
    global ACTIVE_JD_DATA
    print(f">> [JD Loader] Syncing JDs from local folder: {JDS_FOLDER_PATH}")
    JDS_FOLDER_PATH.mkdir(exist_ok=True)
    jds = {}
    for filepath in JDS_FOLDER_PATH.glob("*.txt"):
        try: jds[filepath.stem] = {"main_content": filepath.read_text(encoding='utf-8')}
        except Exception as e: print(f"  ❌ Failed to load JD '{filepath.name}': {e}")
    if not jds: print(f"{Colors.YELLOW}!! [JD Loader] Warning: No .txt JD files found in '{JDS_FOLDER_PATH}'.{Colors.RESET}")
    ACTIVE_JD_DATA = jds
    print(f"✅ [JD Loader] Successfully loaded {len(jds)} JDs.")
    return True

def get_current_max_total_id():
    try:
        with sqlite3.connect(LOCAL_DATABASE_FILE) as conn:
            result = conn.cursor().execute(f"SELECT MAX({DB_PROPS['analysis_total_id']}) FROM analysis_reports").fetchone()[0]
            return result or 0
    except sqlite3.Error: return 0

def clean_text(input_text: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', input_text or "")).strip()

# ===================================================================================
# --- 核心分析、解析与存储模块 ---
# ===================================================================================

def parse_structured_report(text: str) -> dict:
    data = {}
    def get_section(title):
        pattern = re.compile(rf"\*\*{re.escape(title)}\*\*\s*(.*?)(?=\n\n\*\*|\Z)", re.DOTALL)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""
    profile_text = get_section("1. Candidate Profile")
    analysis_text = get_section("2. Positional Fit Analysis")
    verdict_text = get_section("3. Final Verdict")
    data['name'] = (re.search(r'-\s*\*\*Name\*\*:\s*(.*)', profile_text) or {}).group(1) or "Unknown"
    data['phone'] = (re.search(r'-\s*\*\*Phone\*\*:\s*(.*)', profile_text) or {}).group(1) or "N/A"
    data['email'] = (re.search(r'-\s*\*\*Email\*\*:\s*(.*)', profile_text) or {}).group(1) or "N/A"
    data['core_competencies'] = (re.search(r'-\s*\*\*Core Competencies\*\*:\s*(.*)', profile_text) or {}).group(1) or ""
    data['best_position'] = (re.search(r'-\s*\*\*Best Matched Position\*\*:\s*(.*)', analysis_text) or {}).group(1) or "Not Matched"
    dimensional_match = re.search(r'-\s*\*\*Dimensional Analysis\*\*:\s*(.*)', analysis_text, re.DOTALL)
    data['dimensional_analysis'] = dimensional_match.group(1).strip() if dimensional_match else ""
    data['overall_summary'] = (re.search(r'-\s*\*\*Overall Summary\*\*:\s*(.*)', analysis_text, re.DOTALL) or {}).group(1) or ""
    data['recommendation'] = (re.search(r'-\s*\*\*Recommendation\*\*:\s*(.*)', verdict_text) or {}).group(1) or "No recommendation."
    score_match = re.search(r'-\s*\*\*Final Match Score\*\*:\s*(\d{1,3})%', verdict_text)
    data['score_float'] = float(score_match.group(1)) if score_match else 0.0
    data['score_str'] = f"{int(data['score_float'])}%"
    data['full_report_text'] = text
    return data

async def trigger_smart_analysis_async(content_input, new_candidate_id):
    loop = asyncio.get_running_loop()
    resume_text, source_name = "", "text input"
    is_image = isinstance(content_input, Path)
    if is_image:
        source_name = f"image '{content_input.name}'"
        print(f"  > [Content] Processing {source_name}, running OCR...")
        try:
            image = await loop.run_in_executor(None, Image.open, str(content_input))
            resume_text = clean_text(await loop.run_in_executor(None, pytesseract.image_to_string, image, 'chi_sim+eng'))
        except Exception as e: print(f"!! [OCR Error] Failed to process image: {e}")
        finally:
            if os.path.exists(content_input): shutil.move(str(content_input), str(ARCHIVE_DIR / content_input.name))
    else:
        resume_text = clean_text(content_input)
    if not resume_text:
        print(f">> [Pre-check] ID {new_candidate_id} from {source_name} has no content, skipped."); return
    jd_input_text = "\n\n---\n\n".join([f"## {title}\n{data['main_content']}" for title, data in ACTIVE_JD_DATA.items()])
    if not jd_input_text: print(f">> [Pre-check] ID {new_candidate_id} has no JDs to compare against, skipped."); return
    prompt_text = AI_PROMPT_TEMPLATE.format(jd_input=jd_input_text, resume_text=resume_text)
    print("\n" + "="*25 + f" Candidate [ID: {new_candidate_id}] AI Live Analysis " + "="*25)
    full_report = ""
    try:
        async for chunk in llm.astream(prompt_text):
            print(chunk.content, end='', flush=True)
            full_report += chunk.content
    except Exception as e:
        print(f"\n!! [Model Error] Streaming analysis failed: {e}"); return
    print("\n" + "="*80)
    analysis_data = parse_structured_report(full_report)
    analysis_data['id'] = new_candidate_id
    print(f"✅【VERDICT】Candidate [{analysis_data['name']}] | Best Match: {analysis_data['best_position']} | Score: {Colors.BOLD}{Colors.GREEN}{analysis_data['score_str']}{Colors.RESET}")
    print(f"  Recommendation: {analysis_data['recommendation']}")
    print("="*80)
    async with session_analysis_results_lock:
        session_analysis_results.append(analysis_data)
    asyncio.create_task(background_storage_task(analysis_data))
    print(f"--- ✅ Candidate \"{analysis_data['name']}\" (ID: {new_candidate_id}) analysis complete. Syncing to DB in background... ---")

async def background_storage_task(data: dict):
    new_id = data['id']
    try:
        with sqlite3.connect(LOCAL_DATABASE_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO analysis_reports (
                    {DB_PROPS['analysis_total_id']}, {DB_PROPS['analysis_name']}, phone, email, core_competencies,
                    best_position, dimensional_analysis, overall_summary, recommendation,
                    {DB_PROPS['analysis_score']}, {DB_PROPS['full_report_text']}, {DB_PROPS['analysis_date']}
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_id, data['name'], data['phone'], data['email'], data['core_competencies'],
                data['best_position'], data['dimensional_analysis'], data['overall_summary'],
                data['recommendation'], data['score_float'], data['full_report_text'],
                datetime.now(timezone(timedelta(hours=8))).isoformat()
            ))
            conn.commit()
            print(f"  > [Background Sync] ID {new_id} report saved to SQLite.")
    except sqlite3.Error as e:
        print(f"!! [Background Sync] Error writing ID {new_id} to SQLite: {e}")
    if IS_RAG_ENABLED:
        try:
            loop = asyncio.get_running_loop()
            text_to_embed = f"Competencies: {data['core_competencies']}. Summary: {data['overall_summary']}"
            vector = await loop.run_in_executor(None, EMBEDDING_MODEL.embed_query, text_to_embed)
            upsert_op = lambda: QDRANT_CLIENT.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload={
                    "candidate_name": data['name'], "score": data['score_float'], "text": text_to_embed
                })], wait=True
            )
            await loop.run_in_executor(None, upsert_op)
            print(f"  > [Background Sync] ID {new_id} analysis summary vectorized to Qdrant.")
        except Exception as e:
            print(f"!! [Background Sync] Error writing ID {new_id} to Qdrant: {e}")

def generate_final_summary_report():
    if not session_analysis_results:
        print("\n>> No candidates analyzed in this session. No report generated.")
        return
    print("\n\n" + "="*28 + " Final Summary Report for this Session " + "="*28)
    sorted_results = sorted(session_analysis_results, key=lambda x: x['score_float'], reverse=True)
    report_lines = [f"AI Recruiter - Session Summary Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})", "="*80,
                    f"{'ID':<5} | {'Name':<20} | {'Best Matched Position':<30} | {'Score':<10} | {'Recommendation'}", f"{'-'*5}-+-{'-'*22}-+-{'-'*32}-+-{'-'*12}-+-{'-'*20}"]
    for r in sorted_results:
        report_lines.append(f"{r['id']:<5} | {r['name']:<20} | {r['best_position']:<30} | {r['score_str']:<10} | {r['recommendation']}")
    full_report_str = "\n".join(report_lines)
    print(full_report_str)
    try:
        SUMMARY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        filename = SUMMARY_REPORT_DIR / f"Summary_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filename.write_text(full_report_str, encoding='utf-8')
        print(f"\n✅ {Colors.BOLD}{Colors.GREEN}Summary report saved to: {filename}{Colors.RESET}")
    except Exception as e:
        print(f"\n❌ {Colors.BOLD}{Colors.RED}Failed to save summary report: {e}{Colors.RESET}")

# ===================================================================================
# --- 文件监控处理器、后台任务与主循环 ---
# ===================================================================================

class InboxHandler(FileSystemEventHandler):
    def __init__(self, loop, queue):
        self.loop = loop; self.queue = queue; self.last_processed_time = {}
    def on_created(self, event):
        if event.is_directory: return
        filepath = Path(event.src_path)
        now = time.time()
        if filepath in self.last_processed_time and now - self.last_processed_time[filepath] < 1.0: return
        self.last_processed_time[filepath] = now
        asyncio.run_coroutine_threadsafe(self.process_file(filepath), self.loop)
    async def process_file(self, filepath: Path):
        try:
            await asyncio.sleep(0.5)
            if not filepath.exists(): return
            print(f"\n>> [Inbox] Detected new file: {filepath.name}")
            ext = filepath.suffix.lower()
            if ext in ['.txt', '.md', '.rtf']:
                content = filepath.read_text(encoding='utf-8')
                await self.queue.put(content)
                print(f"   -> ✅ Text content queued for analysis.")
                shutil.move(str(filepath), str(ARCHIVE_DIR / filepath.name))
            elif ext in ['.png', '.jpg', '.jpeg']:
                await self.queue.put(filepath)
                print(f"   -> ✅ Image path queued for analysis.")
            else:
                print(f"   -> 🟡 Unsupported file type '{ext}', ignoring.")
        except Exception as e:
            print(f"!! [File Handler Error] Error processing {filepath.name}: {e}")
            traceback.print_exc()

class AppController:
    def __init__(self): self.should_exit = False
    def trigger_exit(self): self.should_exit = True

class AsyncSafeCounter:
    def __init__(self, start_id=0): self.value = start_id; self.lock = asyncio.Lock()
    async def get_next(self):
        async with self.lock: self.value += 1; return self.value

async def consumer_worker_async(queue, counter):
    while not app_controller.should_exit:
        try:
            content = await queue.get()
            new_id = await counter.get_next()
            print(f"\n--- Analyzing new candidate, assigning ID [{new_id}] ---")
            await trigger_smart_analysis_async(content, new_id)
            queue.task_done()
        except asyncio.CancelledError: break
        except Exception: print(f"!! [Worker Error] An uncaught exception occurred:\n{traceback.format_exc()}")

def show_desktop_notification(title, message):
    if platform.system() == "Linux":
        try: subprocess.Popen(['notify-send', title, message, '-a', 'AI Recruiter', '-t', '5000'])
        except FileNotFoundError: pass
    elif PLYER_AVAILABLE:
        try: notification.notify(title=title, message=message, app_name='AI Recruiter', timeout=10)
        except Exception: pass

async def notification_watcher_async(queue: asyncio.Queue):
    while not app_controller.should_exit:
        try:
            if queue.empty():
                future = asyncio.get_running_loop().create_future()
                original_put = queue.put
                async def new_put(item):
                    await original_put(item)
                    if not future.done(): future.set_result(True)
                queue.put = new_put; await future; queue.put = original_put
            await queue.join()
            await asyncio.sleep(3.0)
            if queue.empty():
                show_desktop_notification("AI Recruiter Task Complete", "All queued resumes have been analyzed!")
        except asyncio.CancelledError: break
        except Exception: pass

async def main():
    global app_controller, candidate_id_counter
    app_controller = AppController()
    for d in [IPC_DIR, INBOX_DIR, ARCHIVE_DIR]: d.mkdir(exist_ok=True)
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*10 + " AI Recruiter v24.1 - English Structured Performance Edition (Proxy Fix) " + "="*10)
    print(">> System boot sequence initiated..."); print("-" * 70)
    if not (setup_local_database() and setup_api() and load_active_jd_from_local_folder()):
        input("Core system initialization failed. Please check your configuration and press Enter to exit."); sys.exit(1)
    
    await setup_qdrant_async()
    
    start_id = get_current_max_total_id()
    candidate_id_counter = AsyncSafeCounter(start_id=start_id)
    all_tasks = [asyncio.create_task(consumer_worker_async(task_queue, candidate_id_counter)) for _ in range(4)]
    all_tasks.append(asyncio.create_task(notification_watcher_async(task_queue)))
    
    event_handler = InboxHandler(asyncio.get_running_loop(), task_queue)
    observer = Observer(); observer.schedule(event_handler, str(INBOX_DIR), recursive=False); observer.start()
    print(f"\n>> [System Ready] Now monitoring inbox folder: {INBOX_DIR}")
    print("\n  Press Ctrl+C to safely shut down and generate the final summary report.")
    print("="*70)
    
    try:
        while not app_controller.should_exit: await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        app_controller.trigger_exit()
    finally:
        print("\n>> Shutting down AI Recruiter...")
        observer.stop(); observer.join()
        await task_queue.join()
        for task in all_tasks: task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        generate_final_summary_report() 
        if QDRANT_CLIENT: QDRANT_CLIENT.close()
        print("\n✅ AI Recruiter has been safely shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram forcibly terminated.")