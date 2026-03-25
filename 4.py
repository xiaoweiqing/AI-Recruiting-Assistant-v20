# ===================================================================================
#      AI 简历信息提取助理 v1.1 (纯 Ollama 架构版)
# ===================================================================================
# v1.1 更新:
# - [架构统一] 移除 SentenceTransformer 依赖，所有 AI 计算 (提取 & 向量化)
#   全部通过 Ollama 服务进行，简化技术栈。
# - [模型统一] 向量化任务明确使用 'embeddinggemma' 模型。
# ===================================================================================

import time
import sys
import os
import re
import json
import notion_client
from datetime import datetime, timezone, timedelta
import shutil
import hashlib
import traceback
import asyncio
from dotenv import load_dotenv

# --- 核心依赖 ---
# 【【【 修改点 】】】 引入 OllamaEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient, models

# --- 加载环境变量 ---
load_dotenv()

# ==============================================================================
# ⬇⬇⬇ 0. 用户配置区 (保持不变) ⬇⬇⬇
# ==============================================================================
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
CANDIDATE_DB_ID = "22a584b1cda3802dbb5dd8e1aba9a967"
CANDIDATE_PROFILE_HUB_DB_ID = "238584b1cda380888c78ee2ca403cd72"
TRAINING_HUB_DATABASE_ID = "231584b1cda380a1927be2ab6f22cf33"

NOTION_PROPS = {
    "analysis_name": "候选人姓名", "analysis_date": "分析日期", "analysis_source": "源文件名",
    "analysis_phone": "联系电话", "analysis_email": "候选人邮箱", "analysis_reason": "评分理由",
    "profile_name": "姓名", "profile_relation_to_analysis": "关联分析报告",
    "profile_core_skills": "核心能力或技能", "profile_experience": "岗位经验",
    "profile_resume_hash": "简历内容哈希",
    "training_task_title": "训练任务", "training_task_type": "任务类型", "training_input": "源数据 (Input)",
    "training_output": "理想输出 (Output)", "training_status": "标注状态", "training_relation_to_candidate": "源链接-候选人中心",
}

LLM = None
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_RAG_ENABLED = False
# 【【【 重要提示 】】】
# 因为更换了Embedding模型，向量格式会不兼容。
# 强烈建议为新架构使用一个全新的、独立的 Collection 名称，避免数据冲突。
RECRUITMENT_COLLECTION_NAME = "ai_recruitment_assistant_embeddinggemma"
WORKER_COUNT = 4

# ==============================================================================
# ⬇⬇⬇ AI 信息提取 (Prompt & 解析 - 保持不变) ⬇⬇⬇
# ==============================================================================
EXTRACT_PROMPT_V1 = """
TASK: You are an ultra-fast resume information extractor. Extract key information from the RESUME provided.
RULES:
1. Be extremely fast and concise.
2. Your entire output MUST follow this exact format.
3. For SKILLS, provide a comma-separated list of key technologies and skills.
4. For SUMMARY, provide a single, neutral sentence summarizing the candidate's core experience.
5. For COMPANIES, list the primary companies the candidate has worked for.
---
NAME: [Candidate's Name]
EMAIL: [Candidate's Email or N/A]
PHONE: [Candidate's Phone or N/A]
SKILLS: [Comma-separated skills, e.g., Python, Java, Docker, Agile]
SUMMARY: [A single sentence summary of their experience]
COMPANIES: [Company A, Company B, Company C]
---

RESUME:
{resume_text}
"""

def parse_extraction_output(text):
    data = {}
    fields = ["NAME", "EMAIL", "PHONE", "SKILLS", "SUMMARY", "COMPANIES"]
    for i, field in enumerate(fields):
        next_field = fields[i+1] if i + 1 < len(fields) else None
        pattern = f"{field}:(.*?)" + (f"(?=\n{next_field}:)" if next_field else "$")
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            key = field.lower()
            if key == "companies":
                data[key] = [c.strip() for c in value.replace('[','').replace(']','').split(',') if c.strip()]
            else:
                data[key] = value
    return data

# ==============================================================================
# ⬇⬇⬇ 系统设置 & 核心辅助函数 ⬇⬇⬇
# ==============================================================================
def setup_qdrant_and_embedding():
    """
    【【【 已修改为使用 OllamaEmbeddings 】】】
    """
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED
    try:
        print(">> [知识库] 正在初始化 Ollama embedding 模型 (embeddinggemma)...")
        # 直接使用 LangChain 的 OllamaEmbeddings
        EMBEDDING_MODEL = OllamaEmbeddings(model="embeddinggemma")
        
        # 测试一下获取向量维度
        test_vector = EMBEDDING_MODEL.embed_query("test")
        vector_size = len(test_vector)
        print(f">> [知识库] 向量维度: {vector_size}。")

        print(">> [知识库] 正在连接到中央数据库枢纽 (localhost:6333)...")
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
        
        collections_response = QDRANT_CLIENT.get_collections()
        if RECRUITMENT_COLLECTION_NAME not in [c.name for c in collections_response.collections]:
            print(f">> [知识库] 集合 '{RECRUITMENT_COLLECTION_NAME}' 不存在，正在创建...")
            QDRANT_CLIENT.recreate_collection(
                collection_name=RECRUITMENT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )

        print("✅ [知识库] 成功连接！")
        IS_RAG_ENABLED = True
    except Exception as e:
        print(f"❌ [知识库] 初始化失败: {e}")
        IS_RAG_ENABLED = False

def configure_llm():
    global LLM
    try:
        print(">> [AI大脑] 初始化本地 LLM (Ollama/gemma3:4b)...")
        LLM = ChatOllama(model="gemma3:4b", temperature=0.0)
        LLM.invoke("Hi")
        print("✅ [AI大脑] 已准备就绪。")
        return True
    except Exception as e:
        print(f"❌ [AI大脑] 初始化失败: {e}")
        return False

# 其他辅助函数保持不变
def read_file_content(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            import fitz
            with fitz.open(file_path) as doc: return "".join(page.get_text() for page in doc), None
        elif ext == '.docx':
            import docx
            return "\n".join([p.text for p in docx.Document(file_path).paragraphs]), None
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read(), None
        else: return None, "Unsupported file type"
    except Exception as e: return None, str(e)

def get_content_hash(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

def split_text_for_notion(text, chunk_size=1999):
    if not text or not isinstance(text, str): return [{"type": "text", "text": {"content": ""}}]
    clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return [{"type": "text", "text": {"content": clean_text[i:i + chunk_size]}} for i in range(0, len(clean_text), chunk_size)]


# ==============================================================================
# ⬇⬇⬇ Notion 交互函数 (保持不变) ⬇⬇⬇
# ==============================================================================
async def check_candidate_existence_async(async_notion, candidate_name, candidate_email):
    # ... (此函数代码不变)
    if not CANDIDATE_PROFILE_HUB_DB_ID: return None
    db_filter = {"or": []}
    if candidate_email and candidate_email.lower() != 'n/a':
        db_filter["or"].append({"property": NOTION_PROPS["analysis_email"], "email": {"equals": candidate_email}})
    if candidate_name and candidate_name not in ['N/A', '未知候选人']:
        db_filter["or"].append({"property": NOTION_PROPS["profile_name"], "title": {"equals": candidate_name}})
    if not db_filter["or"]: return None
    try:
        response = await async_notion.databases.query(database_id=CANDIDATE_PROFILE_HUB_DB_ID, filter=db_filter, page_size=1)
        if response and response['results']:
            page = response['results'][0]
            hash_prop = page['properties'].get(NOTION_PROPS["profile_resume_hash"], {})
            old_hash = hash_prop.get('rich_text', [{}])[0].get('plain_text', '') if hash_prop.get('rich_text') else ''
            analysis_relation = page['properties'].get(NOTION_PROPS["profile_relation_to_analysis"], {}).get('relation', [])
            analysis_page_id = analysis_relation[0]['id'] if analysis_relation else None
            return {"profile_page_id": page['id'], "analysis_page_id": analysis_page_id, "old_hash": old_hash}
    except Exception: return None
    return None

async def save_to_notion_async(async_notion, report, resume_text, existing_info):
    # ... (此函数代码不变)
    name = report.get('name', 'N/A')
    summary = report.get('summary', 'N/A')
    skills = report.get('skills', 'N/A')
    beijing_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    analysis_props = {
        NOTION_PROPS["analysis_name"]: {"title": [{"text": {"content": name}}]},
        NOTION_PROPS["analysis_email"]: {"email": report.get('email') if report.get('email', 'n/a').lower() != 'n/a' else None},
        NOTION_PROPS["analysis_phone"]: {"phone_number": report.get('phone') if report.get('phone', 'n/a').lower() != 'n/a' else None},
        NOTION_PROPS["analysis_date"]: {"date": {"start": beijing_time.isoformat()}},
        NOTION_PROPS["analysis_source"]: {"rich_text": [{"text": {"content": report.get('filename', 'N/A')}}]},
        NOTION_PROPS["analysis_reason"]: {"rich_text": split_text_for_notion(f"AI摘要: {summary}")}
    }
    analysis_page_id = None
    try:
        if existing_info and existing_info.get("analysis_page_id"):
            analysis_page_id = existing_info["analysis_page_id"]
            await async_notion.pages.update(page_id=analysis_page_id, properties=analysis_props)
        else:
            new_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_DB_ID}, properties=analysis_props)
            analysis_page_id = new_page.get('id')
    except Exception as e:
        print(f"   !! [Notion] 操作分析中心失败: {e}")
        return None, None
    profile_props = {
        NOTION_PROPS["profile_name"]: {"title": [{"text": {"content": name}}]},
        NOTION_PROPS["profile_core_skills"]: {"rich_text": split_text_for_notion(skills)},
        NOTION_PROPS["profile_experience"]: {"rich_text": split_text_for_notion(summary)},
        NOTION_PROPS["profile_resume_hash"]: {"rich_text": [{"text": {"content": get_content_hash(resume_text)}}]},
        NOTION_PROPS["profile_relation_to_analysis"]: {"relation": [{"id": analysis_page_id}]}
    }
    profile_page_id = None
    try:
        if existing_info and existing_info.get("profile_page_id"):
            profile_page_id = existing_info["profile_page_id"]
            await async_notion.pages.update(page_id=profile_page_id, properties=profile_props)
        else:
            new_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_PROFILE_HUB_DB_ID}, properties=profile_props)
            profile_page_id = new_page.get('id')
    except Exception as e: print(f"   !! [Notion] 操作候选人信息库失败: {e}")
    async def write_to_training_hub():
        if not TRAINING_HUB_DATABASE_ID: return
        try:
            output_str = json.dumps(report, ensure_ascii=False, indent=2)
            title = f"【简历提取】{name}"
            props = {
                NOTION_PROPS["training_task_title"]: {"title": [{"text": {"content": title}}]},
                NOTION_PROPS["training_task_type"]: {"select": {"name": "简历快速提取"}},
                NOTION_PROPS["training_input"]: {"rich_text": split_text_for_notion(resume_text)},
                NOTION_PROPS["training_output"]: {"rich_text": split_text_for_notion(output_str)},
                NOTION_PROPS["training_status"]: {"select": {"name": "待审核"}},
                NOTION_PROPS["training_relation_to_candidate"]: {"relation": [{"id": analysis_page_id}]}
            }
            await async_notion.pages.create(parent={"database_id": TRAINING_HUB_DATABASE_ID}, properties=props)
        except Exception as e: print(f"   !! [Notion] 写入训练中心失败: {e}")
    asyncio.create_task(write_to_training_hub())
    print(f"✅ [Notion] 候选人 '{name}' 的所有档案已同步。")
    return analysis_page_id, profile_page_id

async def vectorize_and_store_async(page_id, report):
    """
    【【【 已修改为使用 OllamaEmbeddings 的异步方法 】】】
    """
    if not IS_RAG_ENABLED or not page_id: return
    name = report.get('name', 'N/A')
    try:
        text_to_embed = f"候选人: {name}\n核心技能: {report.get('skills', '')}\n经验摘要: {report.get('summary', '')}"
        
        # OllamaEmbeddings 已经有 aembed_query 异步方法，无需再用 Executor
        vector = await EMBEDDING_MODEL.aembed_query(text_to_embed)
        
        payload = {"summary_text": text_to_embed, "source_title": f"候选人档案: {name}"}
        
        # QdrantClient 是异步的，但 upsert 不是，所以最好还是放在 to_thread 中
        await asyncio.to_thread(
            QDRANT_CLIENT.upsert,
            collection_name=RECRUITMENT_COLLECTION_NAME,
            points=[models.PointStruct(id=str(page_id), vector=vector, payload=payload)],
            wait=True
        )
        print(f"   -> [向量化] '{name}' 的档案已向量化并存入知识库。")
    except Exception as e:
        print(f"   !! [向量化] 失败: {e}")

# ==============================================================================
# ⬇⬇⬇ 主执行流程 (保持不变) ⬇⬇⬇
# ==============================================================================
async def process_single_resume_async(file_path, worker_name):
    # ... (此函数代码不变)
    filename = os.path.basename(file_path)
    async_notion = notion_client.AsyncClient(auth=NOTION_TOKEN)
    loop = asyncio.get_running_loop()
    resume_text, error_msg = await loop.run_in_executor(None, read_file_content, file_path)
    if error_msg or not resume_text or len(resume_text.strip()) < 50: return "skipped_empty"
    prompt = EXTRACT_PROMPT_V1.format(resume_text=resume_text[:8000])
    print(f"\n>> [{worker_name}] 正在从 '{filename}' 提取核心信息...")
    print("   " + "="*20 + " AI提取中 " + "="*20)
    full_response_content = ""
    async for chunk in LLM.astream(prompt):
        print(chunk.content, end="", flush=True)
        full_response_content += chunk.content
    print("\n   " + "="*20 + " 提取完毕 " + "="*22)
    extracted_data = parse_extraction_output(full_response_content)
    if not extracted_data.get("name"): raise Exception(f"AI未能从 '{filename}' 提取出有效的候选人姓名。")
    extracted_data["filename"] = filename
    print(f"\n✅ [{worker_name}] 提取成功: {extracted_data.get('name')}")
    existing_info = await check_candidate_existence_async(async_notion, extracted_data.get('name'), extracted_data.get('email'))
    if existing_info and existing_info.get("old_hash") == get_content_hash(resume_text): return "skipped_no_change"
    analysis_page_id, profile_page_id = await save_to_notion_async(async_notion, extracted_data, resume_text, existing_info)
    page_id_for_vector = profile_page_id if profile_page_id else analysis_page_id
    if page_id_for_vector: asyncio.create_task(vectorize_and_store_async(page_id_for_vector, extracted_data))
    return {"status": "success", "report": extracted_data}

async def resume_worker(name, queue, result_lists):
    # ... (此函数代码不变)
    processed_files, failed_files, skipped_files = result_lists
    while True:
        try:
            file_path = await queue.get()
            filename = os.path.basename(file_path)
            PROCESSED_DIR, FAILED_DIR, SKIPPED_DIR = "processed_resumes", "processed_failed", "processed_skipped"
            try:
                result = await process_single_resume_async(file_path, name)
                if isinstance(result, str) and result.startswith("skipped"):
                    skipped_files.append(f"{filename} (原因: {result.split('_')[-1]})")
                    shutil.move(file_path, os.path.join(SKIPPED_DIR, filename))
                elif isinstance(result, dict) and result.get("status") == "success":
                    processed_files.append(f"{filename} (候选人: {result['report'].get('name')})")
                    shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
            except Exception as e:
                print(f"!! [{name}] 处理 {filename} 严重失败: {e}\n{traceback.format_exc()}");
                failed_files.append(f"{filename} (错误: {str(e)[:50]}...)")
                shutil.move(file_path, os.path.join(FAILED_DIR, filename))
            finally: queue.task_done()
        except asyncio.CancelledError: break

async def batch_mode_high_concurrency():
    # ... (此函数代码不变)
    print("\n" + "="*14 + " 批量简历提取模式 " + "="*14)
    RESUMES_DIR = "resumes_to_process"
    DIRS_TO_CREATE = ["processed_resumes", "processed_failed", "processed_skipped", RESUMES_DIR]
    for d in DIRS_TO_CREATE: os.makedirs(d, exist_ok=True)
    resumes_to_process = [os.path.join(RESUMES_DIR, f) for f in os.listdir(RESUMES_DIR) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    if not resumes_to_process: print(f"\n在 '{RESUMES_DIR}' 文件夹中没有找到简历。"); return
    print(f"\n找到 {len(resumes_to_process)} 份简历, 准备开始高速提取...")
    queue = asyncio.Queue(); result_lists = ([], [], [])
    for file_path in resumes_to_process: await queue.put(file_path)
    worker_tasks = [asyncio.create_task(resume_worker(f"Worker-{i+1}", queue, result_lists)) for i in range(WORKER_COUNT)]
    await queue.join()
    for task in worker_tasks: task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    processed_files, failed_files, skipped_files = result_lists
    print("\n" + "="*70 + "\n✅ 批量提取完毕！\n" + "="*28 + " 最终总结报告 " + "="*28)
    print(f"\n成功处理: {len(processed_files)} 份"); [print(f"  - {f}") for f in processed_files]
    print(f"\n跳过处理: {len(skipped_files)} 份"); [print(f"  - {f}") for f in skipped_files]
    print(f"\n处理失败: {len(failed_files)} 份"); [print(f"  - {f}") for f in failed_files]
    print("="*70)

# ==============================================================================
# ⬇⬇⬇ 主程序入口 (保持不变) ⬇⬇⬇
# ==============================================================================
async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*10 + " AI 简历信息提取助理 v1.1 (纯 Ollama 架构) " + "="*10)
    print(">> 系统启动中...")
    print("-" * 70)
    if not configure_llm(): sys.exit(1)
    setup_qdrant_and_embedding()
    await batch_mode_high_concurrency()

if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except ImportError: pass
    try:
        asyncio.run(main())
    except KeyboardInterrupt: print("\n>> [系统] 用户中断，正在关闭...")
    except Exception as e:
        print("\n" + "="*20 + " 发生致命错误! " + "="*20)
        print(f"!! 错误: {e}"); traceback.print_exc()
        input("\n程序已崩溃，按回车键退出。")
    finally:
        print("\n>> [系统] 程序执行完毕。")
        input("按回车键退出...")