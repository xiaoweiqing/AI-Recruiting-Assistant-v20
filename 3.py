# ===================================================================================
#      AI 招聘助理 v22.0 - 火箭瘦身版 (Rocket Edition)
# ===================================================================================
# 版本亮点:
# - 【速度至上】: 彻底移除所有影响速度的非核心功能，包括Google搜索、公司研究、数据增强。
# - 【超轻Prompt】: 采用全新的、极度简化的英文Prompt，不再强制JSON输出，大幅降低模型负担。
# - 【流程重构】: 核心处理逻辑被完全重写，专注“快速匹配->上传Notion->向量化”三大核心任务。
# - 【依赖最小化】: 移除了google-api-python-client的必要性。
# ===================================================================================

import time
from datetime import datetime, timezone, timedelta
import sys
import os
import re
import json
import notion_client
import shutil
import hashlib
import traceback
import asyncio

# 核心依赖
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# ⬇⬇⬇ 0. 用户配置区 (已瘦身) ⬇⬇⬇
# ==============================================================================
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
JD_HUB_DATABASE_ID = os.getenv("JD_HUB_DATABASE_ID")
CANDIDATE_DB_ID = os.getenv("CANDIDATE_DB_ID")
CANDIDATE_PROFILE_HUB_DB_ID = os.getenv("CANDIDATE_PROFILE_HUB_DB_ID")

# Notion 属性名 (仅保留核心)
NOTION_PROPS = {
    "analysis_name": "候选人姓名", "analysis_score": "匹配度评分", "analysis_reason": "评分理由",
    "analysis_date": "分析日期", "analysis_source": "源文件名", "analysis_phone": "联系电话",
    "analysis_email": "候选人邮箱", "analysis_priority": "优先级", "analysis_best_fit_position": "最匹配职位",
    "jd_status": "状态", "jd_title": "职位名称",
    "profile_name": "姓名", "profile_relation_to_analysis": "关联分析报告",
    "profile_resume_hash": "简历内容哈希",
}

# 系统配置
ACTIVE_JD_DATA = {}
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
LLM = None
IS_RAG_ENABLED = False
RECRUITMENT_COLLECTION_NAME = "ai_recruitment_assistant_embeddinggemma_v22_rocket" # 使用新集合
WORKER_COUNT = 1

# ==============================================================================
# ⬇⬇⬇ AI分析 & Prompting (火箭版) ⬇⬇⬇
# ==============================================================================
ROCKET_PROMPT_V1 = """
TASK: Act as an ultra-fast AI recruiting assistant. Analyze the RESUME against the JOB DESCRIPTION.
RULES:
1.  Be extremely fast and concise.
2.  Your entire output MUST follow this exact format, with each value on a new line. Do not add any extra text.
---
NAME: [Candidate's Name]
EMAIL: [Candidate's Email or N/A]
PHONE: [Candidate's Phone or N/A]
MATCH_SCORE: [A score from 1 to 10, judging the match]
REASON: [A single, brief sentence in English explaining the score]
---

JOB DESCRIPTION:
{jd_text}
---
RESUME:
{resume_text}
"""

def parse_rocket_output(text):
    """
    【【【 已升级为更健壮的解析器 v2.0 】】】
    一个更健壮的解析器，能处理前后多余的文本，并更稳定地提取信息。
    """
    data = {}
    # 使用正则表达式从可能混杂的文本中提取核心信息
    # re.DOTALL 让 . 可以匹配换行符, re.IGNORECASE 忽略大小写
    core_match = re.search(r"NAME:.*REASON:.*", text, re.DOTALL | re.IGNORECASE)
    
    if not core_match:
        return data # 如果连核心格式都找不到，直接返回空字典

    core_text = core_match.group(0)
    lines = core_text.strip().split('\n')
    
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            # 兼容 "MATCH SCORE" 这种 key
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()

            if "name" in key: data["name"] = value
            elif "email" in key: data["email"] = value
            elif "phone" in key: data["phone"] = value
            elif "match_score" in key:
                # 使用正则更安全地提取数字
                score_found = re.search(r'\d+', value)
                if score_found:
                    data["score"] = int(score_found.group())
            elif "reason" in key: data["reason"] = value
            
    return data

# ==============================================================================
# ⬇⬇⬇ 核心辅助函数 (已瘦身) ⬇⬇⬇
# ==============================================================================
def setup_qdrant_and_embedding():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED
    try:
        print(">> [知识库] 初始化 Embedding 模型...")
        EMBEDDING_MODEL = OllamaEmbeddings(model="embeddinggemma")
        vector_size = len(EMBEDDING_MODEL.embed_query("test"))
        print(f">> [知识库] 向量维度: {vector_size}。")
        
        print(">> [知识库] 连接到 Qdrant (localhost:6333)...")
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)

        collections_response = QDRANT_CLIENT.get_collections()
        collection_names = [col.name for col in collections_response.collections]
        if RECRUITMENT_COLLECTION_NAME not in collection_names:
            print(f">> [知识库] 集合 '{RECRUITMENT_COLLECTION_NAME}' 不存在，正在创建...")
            QDRANT_CLIENT.recreate_collection(
                collection_name=RECRUITMENT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
        print("✅ [知识库] 连接成功！")
        IS_RAG_ENABLED = True
    except Exception as e:
        print(f"❌ [知识库] 初始化失败: {e}")
        IS_RAG_ENABLED = False

def configure_llm():
    global LLM
    try:
        # 同时修改这里的打印信息和模型名称
        print(">> [AI大脑] 初始化本地 LLM (Ollama/gemma3:4b)...")
        LLM = ChatOllama(model="gemma3:4b", temperature=0.0) # Temperature 设为 0，追求速度和一致性
        LLM.invoke("Hi") 
        print("✅ [AI大脑] 已准备就绪。")
        return True
    except Exception as e:
        print(f"❌ [AI大脑] 初始化失败: {e}")
        return False
async def get_page_content_async(async_notion_client, page_id):
    """这是一个用于深度抓取Notion页面所有文本块内容的辅助函数。"""
    content = []
    try:
        all_blocks = await notion_client.helpers.async_collect_paginated_api(
            async_notion_client.blocks.children.list,
            block_id=page_id
        )
        for block in all_blocks:
            block_type = block.get('type')
            if block_type and block_type in block and 'rich_text' in block[block_type]:
                for text_part in block[block_type].get('rich_text', []):
                    content.append(text_part.get('plain_text', ''))
        return "\n".join(content)
    except Exception as e:
        print(f"   !! [深度抓取] 读取页面 {page_id} 内容失败: {e}")
        return ""
async def load_active_jd_from_notion_async():
    global ACTIVE_JD_DATA
    try:
        print(">> [JD加载器] 从Notion同步有效JD...")
        async_notion = notion_client.AsyncClient(auth=NOTION_TOKEN)
        
        active_pages = await notion_client.helpers.async_collect_paginated_api(
            async_notion.databases.query,
            database_id=JD_HUB_DATABASE_ID,
            filter={"property": NOTION_PROPS["jd_status"], "select": {"equals": "Active"}}
        )
        if not active_pages: return False
            
        async def fetch_jd_content(page):
            props = page.get('properties', {})
            title_list = props.get(NOTION_PROPS["jd_title"], {}).get('title', [])
            title = title_list[0].get('plain_text', "Untitled") if title_list else "Untitled"
            
            # 【【【 终极平衡方案：智能提取JD核心信息 】】】
            
            # 1. 提取您在 Notion 中明确填写的“硬性门槛”
            hard_reqs_list = props.get(NOTION_PROPS.get("jd_hard_requirements", "硬性门槛"), {}).get('rich_text', [])
            hard_reqs_text = "".join(t.get('plain_text', '') for t in hard_reqs_list)

            # 2. 抓取页面正文，但只取前面一部分作为“核心职责”
            # 我们假设前 1500 个字符足够概括核心职责
            full_body_text = await get_page_content_async(async_notion, page.get('id'))
            core_responsibilities = full_body_text[:1500]

            # 3. 将两者拼接成一个精简但有效的 JD 核心描述
            # 优先使用硬性要求，因为它更精准
            if hard_reqs_text:
                jd_core_content = f"Core Responsibilities (from main text):\n{core_responsibilities}\n\nHard Requirements (must-haves):\n{hard_reqs_text}"
            else:
                jd_core_content = core_responsibilities

            if jd_core_content.strip(): 
                print(f"  ✅ 智能提取JD核心: {title}")
                return title, {"content": jd_core_content.strip(), "id": page.get('id')}
            
            print(f"  ❌ 无法提取JD核心: {title}")
            return None, None

        tasks = [fetch_jd_content(page) for page in active_pages]
        results = await asyncio.gather(*tasks)
        ACTIVE_JD_DATA = {title: data for title, data in results if title}
        if not ACTIVE_JD_DATA: return False

        print(f"✅ [JD加载器] 成功加载 {len(ACTIVE_JD_DATA)} 个JD。")
        return True
    except Exception as e:
        print(f"❌ [JD加载器] 同步失败: {e}")
        return False

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

# ==============================================================================
# ⬇⬇⬇ Notion 交互函数 (已瘦身) ⬇⬇⬇
# ==============================================================================
async def check_candidate_existence_async(async_notion, candidate_name, candidate_email):
    # 此函数保持不变，用于去重
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
    except Exception:
        return None
    return None

async def save_to_notion_async(async_notion, report, resume_text, existing_info):
    score = report.get('score', 0)
    tag_map = {10: "🌟 S级 (必须拿下)", 9: "🌟 S级 (必须拿下)", 8: "🔥 A级 (重点跟进)", 7: "✅ B级 (符合预期)"}
    tag = tag_map.get(score, "🤔 C级 (待定观察)")
    
    # 1. 创建或更新分析报告
    analysis_props = {
        NOTION_PROPS["analysis_name"]: {"title": [{"text": {"content": report.get('name', 'N/A')}}]},
        NOTION_PROPS["analysis_email"]: {"email": report.get('email') if 'n/a' not in report.get('email','n/a').lower() else None},
        NOTION_PROPS["analysis_phone"]: {"phone_number": report.get('phone') if 'n/a' not in report.get('phone','n/a').lower() else None},
        NOTION_PROPS["analysis_best_fit_position"]: {"rich_text": [{"text": {"content": report.get('best_fit', 'N/A')}}]},
        NOTION_PROPS["analysis_score"]: {"number": score},
        NOTION_PROPS["analysis_reason"]: {"rich_text": [{"text": {"content": report.get('reason', 'N/A')}}]},
        NOTION_PROPS["analysis_date"]: {"date": {"start": datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()}},
        NOTION_PROPS["analysis_priority"]: {"select": {"name": tag}},
        NOTION_PROPS["analysis_source"]: {"rich_text": [{"text": {"content": report.get('filename', 'N/A')}}]},
    }
    analysis_page_id = None
    if existing_info and existing_info.get("analysis_page_id"):
        analysis_page_id = existing_info["analysis_page_id"]
        await async_notion.pages.update(page_id=analysis_page_id, properties=analysis_props)
    else:
        new_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_DB_ID}, properties=analysis_props)
        analysis_page_id = new_page.get('id')

    # 2. 创建或更新个人档案
    profile_props = {
        NOTION_PROPS["profile_name"]: {"title": [{"text": {"content": report.get('name', 'N/A')}}]},
        NOTION_PROPS["profile_resume_hash"]: {"rich_text": [{"text": {"content": get_content_hash(resume_text)}}]},
        NOTION_PROPS["profile_relation_to_analysis"]: {"relation": [{"id": analysis_page_id}]}
    }
    profile_page_id = None
    if existing_info and existing_info.get("profile_page_id"):
        profile_page_id = existing_info["profile_page_id"]
        await async_notion.pages.update(page_id=profile_page_id, properties=profile_props)
    else:
        new_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_PROFILE_HUB_DB_ID}, properties=profile_props)
        profile_page_id = new_page.get('id')
    
    print(f"✅ [Notion] 候选人 '{report.get('name')}' 的档案已同步。")
    return profile_page_id

async def vectorize_profile_async(page_id, report, resume_text):
    if not IS_RAG_ENABLED: return
    try:
        text_to_embed = f"Candidate: {report.get('name')}. Core competency: {report.get('reason')}. Resume snippet: {resume_text[:500]}"
        vector = await EMBEDDING_MODEL.aembed_query(text_to_embed)
        payload = {"summary_text": text_to_embed, "source_title": f"Candidate: {report.get('name')}"}
        
        await asyncio.to_thread(
            QDRANT_CLIENT.upsert,
            collection_name=RECRUITMENT_COLLECTION_NAME,
            points=[models.PointStruct(id=str(page_id), vector=vector, payload=payload)],
            wait=True
        )
        print(f"   -> [向量化] '{report.get('name')}' 的档案已向量化。")
    except Exception as e:
        print(f"   !! [向量化] 失败: {e}")

# ==============================================================================
# ⬇⬇⬇ 主执行流程 (火箭版) ⬇⬇⬇
# ==============================================================================

# ==============================================================================
# ⬇⬇⬇ 主执行流程 (火箭版) -【【【 已修改为流式输出 v2.0 】】】 ⬇⬇⬇
# ==============================================================================
async def process_single_resume_async(file_path, all_jds_data, worker_name):
    filename = os.path.basename(file_path)
    async_notion = notion_client.AsyncClient(auth=NOTION_TOKEN)

    # 1. 读取文件
    resume_text, error_msg = await asyncio.to_thread(read_file_content, file_path)
    if error_msg or not resume_text or len(resume_text.strip()) < 50:
        return "skipped_empty"

    best_match = {"score": 0}
    
    # 定义简历最大字符数，以提高速度和模型遵循指令的准确性
    MAX_RESUME_CHARS = 5000
    truncated_resume = resume_text[:MAX_RESUME_CHARS]
    
    # 2. 循环匹配所有JD
    print(f">> [{worker_name}] 开始对 '{filename}' 进行快速匹配...")
    for jd_title, jd_data in all_jds_data.items():
        prompt = ROCKET_PROMPT_V1.format(jd_text=jd_data['content'], resume_text=truncated_resume)
        
        # --- 核心修改点：从 ainvoke 改为 astream ---
        
        print(f"\n   -> [{worker_name}] 正在为职位 '{jd_title}' 生成分析...")
        print("   " + "="*20 + " AI思考中 " + "="*20)
        
        full_response_content = ""
        # 使用 async for 循环处理流式响应
        async for chunk in LLM.astream(prompt):
            # chunk.content 就是模型生成的文本块
            print(chunk.content, end="", flush=True) # flush=True 确保立即输出
            full_response_content += chunk.content
            
        print("\n   " + "="*20 + " 思考完毕 " + "="*22)
        
        # 后续操作使用拼接好的完整响应
        parsed_data = parse_rocket_output(full_response_content)

        # 在比较前，确保解析出的数据包含 'score' 键
        if parsed_data.get("score", 0) > best_match.get("score", 0):
            best_match = parsed_data
            best_match["best_fit"] = jd_title
    
    # 检查循环后是否得到了有效评分
    if best_match.get("score", 0) == 0:
        # 在抛出异常前打印最后一次的完整响应，方便调试
        print(f"!! [调试信息] AI最终未能解析出有效评分。最后一次的LLM原始输出为:\n---\n{full_response_content}\n---")
        raise Exception("AI未能对任何职位生成有效评分")

    best_match["filename"] = filename
    reason = best_match.get('reason', 'N/A')
    # 在匹配完成后，也打印理由
    print(f"\n>> [{worker_name}] 匹配完成! 最佳职位: '{best_match.get('best_fit', 'N/A')}' (得分: {best_match.get('score', 0)}) | 理由: {reason}")


    # 3. 去重 (这部分代码保持不变)
    existing_info = await check_candidate_existence_async(async_notion, best_match.get('name'), best_match.get('email'))
    if existing_info and existing_info.get("old_hash") == get_content_hash(resume_text):
        return "skipped_no_change"
        
    # 4. 上传 Notion 并向量化 (这部分代码保持不变)
    profile_page_id = await save_to_notion_async(async_notion, best_match, resume_text, existing_info)
    if profile_page_id:
        asyncio.create_task(vectorize_profile_async(profile_page_id, best_match, resume_text))
    
    return {"status": "success", "report": best_match}

async def resume_worker(name, queue, result_lists):
    processed_summaries, failed_files, skipped_files = result_lists
    while True:
        try:
            file_path = await queue.get()
            filename = os.path.basename(file_path)
            print(f"[{name}] 开始处理: {filename}")
            
            PROCESSED_DIR, FAILED_DIR, SKIPPED_DIR = "processed_resumes", "processed_failed", "processed_skipped"
            
            try:
                result = await process_single_resume_async(file_path, ACTIVE_JD_DATA, name)
                
                if isinstance(result, str) and result.startswith("skipped"):
                    skipped_files.append(f"{filename} ({result.split('_')[-1]})")
                    shutil.move(file_path, os.path.join(SKIPPED_DIR, filename))
                elif isinstance(result, dict) and result.get("status") == "success":
                    report = result['report']
                    processed_summaries.append({"name": report['name'], "score": report['score'], "position": report['best_fit']})
                    print(f"✅ [{name}] 候选人 '{report.get('name')}' ({filename}) 分析同步完成。")
                    shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))

            except Exception as e:
                print(f"!! [{name}] 处理 {filename} 严重失败: {e}\n{traceback.format_exc()}");
                failed_files.append(f"{filename} (错误: {str(e)[:50]}...)")
                shutil.move(file_path, os.path.join(FAILED_DIR, filename))
                
            finally:
                queue.task_done()
        except asyncio.CancelledError:
            break

async def batch_mode_high_concurrency():
    print("\n" + "="*14 + " 批量智能匹配模式 (火箭版) " + "="*14)
    RESUMES_DIR = "resumes_to_process"
    DIRS_TO_CREATE = ["processed_resumes", "processed_failed", "processed_skipped", RESUMES_DIR]
    for d in DIRS_TO_CREATE: os.makedirs(d, exist_ok=True)

    resumes_to_process = [os.path.join(RESUMES_DIR, f) for f in os.listdir(RESUMES_DIR) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    if not resumes_to_process:
        print(f"\n在 '{RESUMES_DIR}' 文件夹中没有找到简历。"); return

    print(f"\n找到 {len(resumes_to_process)} 份简历, 将与 {len(ACTIVE_JD_DATA)} 个职位进行匹配...")
    
    queue = asyncio.Queue()
    result_lists = ([], [], []) # (processed, failed, skipped)

    for file_path in resumes_to_process:
        await queue.put(file_path)

    worker_tasks = [asyncio.create_task(resume_worker(f"Worker-{i+1}", queue, result_lists)) for i in range(WORKER_COUNT)]

    await queue.join()

    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    processed_summaries, failed_files, skipped_files = result_lists
    print("\n" + "="*70 + "\n✅ 批量匹配完毕！\n" + "="*28 + " 最终总结报告 " + "="*28)
    print(f"\n成功处理: {len(processed_summaries)} 份")
    if processed_summaries:
        # 【【【 修改这里 】】】
        print(f"| {'候选人姓名':<20} | {'最高分':<8} | {'最匹配职位':<25} | {'评分理由':<40} |")
        print(f"|{'-'*22}|{'-'*10}|{'-'*27}|{'-'*42}|")
        # 别忘了在 resume_worker 中把 reason 也加到 processed_summaries 里
        # report = result['report']
        # processed_summaries.append({"name": report['name'], "score": report['score'], "position": report['best_fit'], "reason": report['reason']})

        for summary in sorted(processed_summaries, key=lambda x: x['score'], reverse=True):
            name = (summary.get('name', 'N/A')[:17] + '...') if len(summary.get('name', 'N/A')) > 20 else summary.get('name', 'N/A')
            pos = (summary.get('position', 'N/A')[:22] + '...') if len(summary.get('position', 'N/A')) > 25 else summary.get('position', 'N/A')
            reason = (summary.get('reason', 'N/A')[:37] + '...') if len(summary.get('reason', 'N/A')) > 40 else summary.get('reason', 'N/A')
            print(f"| {name:<20} | {summary.get('score', 0):<8.1f} | {pos:<25} | {reason:<40} |")
# ==============================================================================
# ⬇⬇⬇ 主程序入口 (火箭版) ⬇⬇⬇
# ==============================================================================
async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*10 + " AI 招聘助理 v22.0 - 火箭瘦身版 " + "="*10)
    print(">> 系统启动中...")
    print("-" * 70)

    if not configure_llm(): sys.exit(1)
    setup_qdrant_and_embedding()
    if not await load_active_jd_from_notion_async(): sys.exit(1)

    while True:
        print("\n" + "="*28 + " 主菜单 " + "="*28)
        print("  1. [批量快配]   - (火箭版) 快速分析简历")
        print("  Q. 退出程序")
        print("="*65)
        choice = input("请输入您的选择 (1 或 Q): ").strip().upper()
        if choice == '1':
            await batch_mode_high_concurrency()
        elif choice == 'Q':
            break
        else:
            print("\n无效输入。")
        input("\n任务完成，按回车返回主菜单...")

if __name__ == "__main__":
    # ==============================================================================
    # ⬇⬇⬇ 最终的“安全模式”启动器 v2.0 ⬇⬇⬇
    # ==============================================================================
    # 这个启动器增加了全局异常捕获，确保任何问题都能被打印出来。
    
    # 尝试安装并使用 uvloop 以提升性能，如果失败则回退到标准库
    try:
        import uvloop
        uvloop.install()
        print(">> [系统] 使用 uvloop 加速事件循环。")
    except ImportError:
        print(">> [系统] 未安装 uvloop，使用标准 asyncio 事件循环。")
        pass

    # 全局异常捕获
    try:
        # 启动主异步函数
        asyncio.run(main())
        
    except KeyboardInterrupt:
        # 捕获用户按 Ctrl+C 的中断
        print("\n>> [系统] 检测到用户中断，正在关闭...")
        
    except Exception as e:
        # 【【【 关键：捕获任何其他在启动或运行中发生的致命错误 】】】
        print("\n" + "="*20 + " 发生了一个致命错误! " + "="*20)
        print(f"!! 错误类型: {type(e).__name__}")
        print(f"!! 错误信息: {e}")
        print("\n--- 详细错误追溯 (Traceback) ---")
        traceback.print_exc()
        print("="*60)
        input("\n程序已崩溃，请根据以上信息排查问题。按回车键退出。")

    finally:
        print(">> [系统] 程序执行完毕，安全退出。")