# ===================================================================================
#      AI 简历信息提取助理 v2.6 (中文优化 & Notion兼容修复版)
# ===================================================================================
# v2.6 更新:
# - [中文优化] AI Prompt 全面中文化，确保提取结果为中文，更符合中文简历语境。
# - [Notion兼容] 修复了因“岗位经验”属性类型不匹配导致的上传失败问题。
# v2.5 更新:
# - [通知优化] 采纳 Fedora/Linux 原生 `notify-send` 命令。
# - [提取升级] 全面重构 AI Prompt，要求模型提取更详细的结构化信息。
# - [Notion归档] 实现强大的双数据库同步逻辑。
# ===================================================================================

# 【【【 在导入任何其他库之前，首先运行此代码块 】】】
import os
import sys
if 'all_proxy' in os.environ: del os.environ['all_proxy']
if 'ALL_PROXY' in os.environ: del os.environ['ALL_PROXY']
# --------------------------------------------------------------------

import time
import re
import json
import notion_client
from datetime import datetime, timezone, timedelta
import shutil
import hashlib
import traceback
import asyncio
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient, models

import platform
import subprocess

load_dotenv()

# ==============================================================================
# ⬇⬇⬇ 0. 用户配置区 (关键！请仔细核对) ⬇⬇⬇
# ==============================================================================
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
CANDIDATE_DB_ID = os.getenv("CANDIDATE_DB_ID")
CANDIDATE_PROFILE_HUB_DB_ID = os.getenv("CANDIDATE_PROFILE_HUB_DB_ID")

NOTION_PROPS = {
    "analysis_name": "候选人姓名", "analysis_date": "分析日期", "analysis_source": "源文件名",
    "analysis_phone": "联系电话", "analysis_email": "候选人邮箱",
    "profile_name": "姓名", "profile_relation_to_analysis": "关联分析报告",

    # 【【【 请重点核对这里的属性名与你的Notion库列名一致 】】】
    "profile_core_skills": "核心能力或技能", 
    "profile_experience_years": "岗位经验", # 注意：此属性在Notion中可以是“数字”或“文本”类型
    "profile_education": "学历",
    "profile_current_location": "目前所在地",
    "profile_expected_salary": "期望薪资",
    "profile_resume_hash": "简历内容哈希",
}

OLLAMA_LLM_MODEL_NAME = "gpt-oss:20b"

LLM = None; EMBEDDING_MODEL = None; QDRANT_CLIENT = None; IS_RAG_ENABLED = False
RECRUITMENT_COLLECTION_NAME = "ai_recruitment_assistant_embeddinggemma_v3"
WORKER_COUNT = 4

# ==============================================================================
# ⬇⬇⬇ AI 信息提取 (Prompt & 解析 - 已全面中文化) ⬇⬇⬇
# ==============================================================================
# 全新的中文版 Prompt
EXTRACT_PROMPT_V2_CN = """
任务：你是一位专业的HR助理。你的任务是从提供的简历文本中，提取详细、结构化的信息。

规则：
1. 你的全部输出【必须】是一个单一、合法的JSON对象。在JSON前后不要包含任何其他文本。
2. 尽可能多地提取信息。如果在简历中找不到某项信息，请为该字段使用 `null` 或空字符串 `""`。不要编造信息。
3. 对于 `work_experience_years`，请提供一个代表相关工作经验总年限的数值。
4. `core_skills` 应该是一个由逗号分隔的关键技术和能力列表。
5. `full_summary` 应该是一个简洁的段落（3-4句话），总结候选人的专业背景、主要成就和职业重点。

JSON输出格式：
{{
  "name": "...",
  "email": "...",
  "phone": "...",
  "education": "例如：西安科技大学，电子商务学士",
  "work_experience_years": 0.0,
  "current_location": "...",
  "expected_salary": "...",
  "core_skills": "产品管理, 跨团队沟通, 需求分析, 交互设计, 用户研究, ...",
  "full_summary": "一个简短的专业总结段落..."
}}

---
待分析的简历文本：
{resume_text}
"""

def parse_json_extraction_output(text: str):
    try:
        match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
        json_str = match.group(1) if match else text.strip()
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"   !! [JSON解析错误] AI返回的不是有效的JSON格式: {e}")
        return None
    except Exception as e:
        print(f"   !! [解析错误] 处理AI输出时发生未知错误: {e}")
        return None

# ==============================================================================
# ⬇⬇⬇ 系统设置 & 核心辅助函数 (功能不变) ⬇⬇⬇
# ==============================================================================
def show_desktop_notification(title, message):
    print(f"\n🔔 [提醒] 准备发送通知: '{title}'")
    if platform.system() == "Linux":
        try:
            subprocess.Popen(['notify-send', title, message, '-a', 'AI简历助理', '-t', '8000'])
            print("✅ [提醒] 通知命令已成功发出。")
        except FileNotFoundError:
            print("!! [提醒] 错误: 'notify-send' 命令未找到。请确认 libnotify 已安装。")
        except Exception as e:
            print(f"!! [提醒] 发送通知时发生未知错误: {e}")
    else:
        print(f"   -> [提醒] 当前系统为 {platform.system()}，跳过 Linux 原生通知。")

async def setup_qdrant_and_embedding():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED;
    try:
        print(">> [知识库] 正在初始化 Ollama embedding 模型 (embeddinggemma)..."); EMBEDDING_MODEL = OllamaEmbeddings(model="embeddinggemma")
        vector_size = await asyncio.to_thread(EMBEDDING_MODEL.embed_query, "test"); print(f">> [知识库] 向量维度: {len(vector_size)}。")
        print(">> [知识库] 正在连接到中央数据库枢纽 (localhost:6333)..."); QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
        collections_response = await asyncio.to_thread(QDRANT_CLIENT.get_collections)
        if RECRUITMENT_COLLECTION_NAME not in [c.name for c in collections_response.collections]:
            print(f">> [知识库] 集合 '{RECRUITMENT_COLLECTION_NAME}' 不存在，正在创建..."); await asyncio.to_thread(QDRANT_CLIENT.recreate_collection, collection_name=RECRUITMENT_COLLECTION_NAME, vectors_config=models.VectorParams(size=len(vector_size), distance=models.Distance.COSINE))
        print("✅ [知识库] 成功连接！"); IS_RAG_ENABLED = True
    except Exception as e: print(f"❌ [知识库] 初始化失败: {e}"); IS_RAG_ENABLED = False

async def configure_llm():
    global LLM;
    try:
        print(f">> [AI大脑] 初始化本地 LLM (Ollama/{OLLAMA_LLM_MODEL_NAME})..."); LLM = ChatOllama(model=OLLAMA_LLM_MODEL_NAME, temperature=0.0)
        print("   -> 正在测试与 Ollama 服务的连接..."); await LLM.ainvoke("Hi"); print("✅ [AI大脑] 已准备就绪。"); return True
    except Exception as e: print(f"❌ [AI大脑] 初始化失败: {e}"); print(f"   -> 请确保 Ollama 服务正在后台运行，并且模型 '{OLLAMA_LLM_MODEL_NAME}' 已经通过 'ollama pull' 成功下载。"); return False

def read_file_content(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower();
        if ext == '.pdf': import fitz; doc = fitz.open(file_path); return "".join(page.get_text() for page in doc), None
        elif ext == '.docx': import docx; return "\n".join([p.text for p in docx.Document(file_path).paragraphs]), None
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read(), None
        else: return None, "Unsupported file type"
    except Exception as e: return None, str(e)

def get_content_hash(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

# ==============================================================================
# ⬇⬇⬇ Notion 交互 & 向量化 (已修复类型兼容问题) ⬇⬇⬇
# ==============================================================================
def format_json_for_notion(data: dict) -> list:
    if not data: return []
    pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
    chunks = [pretty_json[i:i + 1900] for i in range(0, len(pretty_json), 1900)]
    blocks = []
    if chunks:
        blocks.append({"object": "block", "type": "code", "code": {"rich_text": [{"type": "text", "text": {"content": chunks[0]}}], "language": "json"}})
        for chunk in chunks[1:]:
             blocks.append({"object": "block", "type": "paragraph", "paragraph": { "rich_text": [{"type": "text", "text": {"content": chunk}}] }})
    return blocks

async def save_to_notion_async(async_notion, report_data: dict, resume_text: str, filename: str):
    if not report_data: print("   !! [Notion同步] 报告数据为空，跳过同步。"); return None, None
    name = report_data.get('name', '未知候选人'); beijing_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    analysis_page_id, profile_page_id = None, None
    try:
        analysis_props = {
            NOTION_PROPS["analysis_name"]: {"title": [{"text": {"content": name}}]},
            NOTION_PROPS["analysis_email"]: {"email": report_data.get('email')},
            NOTION_PROPS["analysis_phone"]: {"phone_number": report_data.get('phone')},
            NOTION_PROPS["analysis_date"]: {"date": {"start": beijing_time.isoformat()}},
            NOTION_PROPS["analysis_source"]: {"rich_text": [{"text": {"content": filename}}]},
        }
        new_analysis_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_DB_ID}, properties={k: v for k, v in analysis_props.items() if v})
        analysis_page_id = new_analysis_page.get('id')
        json_blocks = format_json_for_notion(report_data)
        if analysis_page_id and json_blocks:
            await async_notion.blocks.children.append(block_id=analysis_page_id, children=json_blocks)
        print(f"✅ [Notion] 已创建分析报告 '{name}' 并归档了完整提取数据。")
    except Exception as e: print(f"   !! [Notion] 操作分析报告库失败: {e}"); traceback.print_exc()
    if analysis_page_id:
        try:
            profile_props = {
                NOTION_PROPS["profile_name"]: {"title": [{"text": {"content": name}}]},
                NOTION_PROPS["profile_relation_to_analysis"]: {"relation": [{"id": analysis_page_id}]},
                NOTION_PROPS["profile_resume_hash"]: {"rich_text": [{"text": {"content": get_content_hash(resume_text)}}]},
            }
            if report_data.get("core_skills"): profile_props[NOTION_PROPS["profile_core_skills"]] = {"rich_text": [{"text": {"content": report_data["core_skills"]}}]}
            if report_data.get("education"): profile_props[NOTION_PROPS["profile_education"]] = {"rich_text": [{"text": {"content": report_data["education"]}}]}
            if report_data.get("current_location"): profile_props[NOTION_PROPS["profile_current_location"]] = {"rich_text": [{"text": {"content": report_data["current_location"]}}]}
            if report_data.get("expected_salary"): profile_props[NOTION_PROPS["profile_expected_salary"]] = {"rich_text": [{"text": {"content": report_data["expected_salary"]}}]}
            
            # --- 【【【 核心修复区 】】】 ---
            # 检查 "work_experience_years" 是否存在
            if report_data.get("work_experience_years") is not None:
                exp_years = report_data["work_experience_years"]
                # 尝试将其作为数字发送
                try:
                    profile_props[NOTION_PROPS["profile_experience_years"]] = {"number": float(exp_years)}
                    new_profile_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_PROFILE_HUB_DB_ID}, properties={k: v for k, v in profile_props.items() if v})
                except (ValueError, TypeError, notion_client.errors.APIResponseError) as e:
                    # 如果作为数字发送失败 (无论是代码转换失败还是Notion API拒绝)
                    if "number" in str(e).lower() or isinstance(e, (ValueError, TypeError)):
                        # 我们就把它作为文本再次尝试发送
                        print("   -> [Notion兼容] '岗位经验' 作为数字上传失败，尝试作为文本上传...")
                        profile_props[NOTION_PROPS["profile_experience_years"]] = {"rich_text": [{"text": {"content": str(exp_years)}}]}
                        new_profile_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_PROFILE_HUB_DB_ID}, properties={k: v for k, v in profile_props.items() if v})
                    else:
                        raise e # 如果是其他不相关的错误，则重新抛出
            else: # 如果没有工作经验字段，正常创建页面
                new_profile_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_PROFILE_HUB_DB_ID}, properties={k: v for k, v in profile_props.items() if v})
                
            profile_page_id = new_profile_page.get('id')
            print(f"✅ [Notion] 已为 '{name}' 创建详细信息档案并自动填充属性。")
        except Exception as e: print(f"   !! [Notion] 操作候选人信息库失败: {e}"); traceback.print_exc()
    return analysis_page_id, profile_page_id

async def vectorize_and_store_async(page_id, report):
    if not IS_RAG_ENABLED or not page_id: return
    name = report.get('name', 'N/A');
    try:
        text_to_embed = f"候选人: {name}\n核心技能: {report.get('core_skills', '')}\n经验摘要: {report.get('full_summary', '')}"
        vector = await EMBEDDING_MODEL.aembed_query(text_to_embed)
        payload = {"summary_text": text_to_embed, "source_title": f"候选人档案: {name}"}
        await asyncio.to_thread(QDRANT_CLIENT.upsert, collection_name=RECRUITMENT_COLLECTION_NAME, points=[models.PointStruct(id=str(page_id), vector=vector, payload=payload)], wait=True)
        print(f"   -> [向量化] '{name}' 的档案已向量化并存入知识库。")
    except Exception as e: print(f"   !! [向量化] 失败: {e}")

# ==============================================================================
# ⬇⬇⬇ 核心处理流程 (已更新) ⬇⬇⬇
# ==============================================================================
async def process_single_resume_async(file_path, worker_name):
    filename = os.path.basename(file_path); loop = asyncio.get_running_loop();
    resume_text, error_msg = await loop.run_in_executor(None, read_file_content, file_path)
    if error_msg or not resume_text or len(resume_text.strip()) < 50: return "skipped_empty"
    prompt = EXTRACT_PROMPT_V2_CN.format(resume_text=resume_text[:8000])
    print(f"\n>> [{worker_name}] 正在从 '{filename}' 深度提取信息..."); print("   " + "="*20 + " AI提取中 " + "="*20); 
    full_response_content = "";
    async for chunk in LLM.astream(prompt):
        print(chunk.content, end="", flush=True)
        full_response_content += chunk.content
    print("\n   " + "="*20 + " 提取完毕 " + "="*22);
    extracted_data = parse_json_extraction_output(full_response_content)
    if not extracted_data or not extracted_data.get("name"): raise Exception(f"AI未能从 '{filename}' 提取出有效的JSON数据或候选人姓名。")
    extracted_data["filename"] = filename; print(f"\n✅ [{worker_name}] 提取成功: {extracted_data.get('name')}");
    async_notion = notion_client.AsyncClient(auth=NOTION_TOKEN)
    analysis_page_id, profile_page_id = await save_to_notion_async(async_notion, extracted_data, resume_text, filename)
    page_id_for_vector = profile_page_id if profile_page_id else analysis_page_id
    if page_id_for_vector: asyncio.create_task(vectorize_and_store_async(page_id_for_vector, extracted_data))
    return {"status": "success", "report": extracted_data}

async def resume_worker(name, queue, result_lists):
    processed_files, failed_files, skipped_files = result_lists;
    while True:
        try:
            file_path = await queue.get(); filename = os.path.basename(file_path);
            PROCESSED_DIR, FAILED_DIR, SKIPPED_DIR = "processed_resumes", "processed_failed", "processed_skipped";
            try:
                result = await process_single_resume_async(file_path, name);
                if isinstance(result, str) and result.startswith("skipped"): skipped_files.append(f"{filename} (原因: {result.split('_')[-1]})"); shutil.move(file_path, os.path.join(SKIPPED_DIR, filename));
                elif isinstance(result, dict) and result.get("status") == "success": processed_files.append(f"{filename} (候选人: {result['report'].get('name')})"); shutil.move(file_path, os.path.join(PROCESSED_DIR, filename));
            except Exception as e: print(f"!! [{name}] 处理 {filename} 严重失败: {e}\n{traceback.format_exc()}"); failed_files.append(f"{filename} (错误: {str(e)[:50]}...)"); shutil.move(file_path, os.path.join(FAILED_DIR, filename));
            finally: queue.task_done()
        except asyncio.CancelledError: break

async def batch_mode_high_concurrency():
    print("\n" + "="*14 + " 批量简历提取模式 " + "="*14); RESUMES_DIR = "resumes_to_process"; DIRS_TO_CREATE = ["processed_resumes", "processed_failed", "processed_skipped", RESUMES_DIR];
    for d in DIRS_TO_CREATE: os.makedirs(d, exist_ok=True);
    resumes_to_process = [os.path.join(RESUMES_DIR, f) for f in os.listdir(RESUMES_DIR) if f.lower().endswith(('.pdf', '.docx', '.txt'))];
    if not resumes_to_process: print(f"\n❌ 在 '{RESUMES_DIR}' 文件夹中没有找到任何简历。请添加文件后再试。"); return;
    print(f"\n找到 {len(resumes_to_process)} 份简历, 准备开始高速提取..."); queue = asyncio.Queue(); result_lists = ([], [], []);
    for file_path in resumes_to_process: await queue.put(file_path);
    worker_tasks = [asyncio.create_task(resume_worker(f"Worker-{i+1}", queue, result_lists)) for i in range(WORKER_COUNT)];
    await queue.join();
    for task in worker_tasks: task.cancel();
    await asyncio.gather(*worker_tasks, return_exceptions=True);
    processed_files, failed_files, skipped_files = result_lists;
    print("\n" + "="*70 + "\n✅ 批量提取完毕！\n" + "="*28 + " 最终总结报告 " + "="*28);
    print(f"\n成功处理: {len(processed_files)} 份"); [print(f"  - {f}") for f in processed_files];
    print(f"\n跳过处理: {len(skipped_files)} 份"); [print(f"  - {f}") for f in skipped_files];
    print(f"\n处理失败: {len(failed_files)} 份"); [print(f"  - {f}") for f in failed_files];
    print("="*70)
    show_desktop_notification("简历处理完成！", f"成功: {len(processed_files)}, 跳过: {len(skipped_files)}, 失败: {len(failed_files)}")

# ==============================================================================
# ⬇⬇⬇ 主程序入口 (功能不变) ⬇⬇⬇
# ==============================================================================
async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*10 + " AI 简历信息提取助理 v2.6 (中文优化 & Notion兼容修复版) " + "="*10)
    print(">> 系统启动中..."); print("-" * 70)
    if not await configure_llm(): sys.exit(1)
    await setup_qdrant_and_embedding()
    print("-" * 70); print("✅ 系统初始化完成！"); await asyncio.sleep(2)
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*28 + " 主 菜 单 " + "="*29); print("\n  [ 1 ]  开始处理 'resumes_to_process' 文件夹中的新简历"); print("\n  [ Q ]  退出程序"); print("\n" + "="*65)
        choice = await asyncio.to_thread(input, "请输入您的选择并按回车: ")
        if choice == '1':
            await batch_mode_high_concurrency()
            print("\n>> [系统] 本轮处理已完成。按回车键返回主菜单...")
            await asyncio.to_thread(input)
        elif choice.lower() == 'q': print("\n>> [系统] 感谢使用，正在退出..."); break
        else: print("\n!! 无效输入，请重新选择。"); await asyncio.sleep(2)

if __name__ == "__main__":
    try: import uvloop; uvloop.install()
    except ImportError: pass
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\n>> [系统] 用户中断，强制退出...")
    except Exception as e: 
        print(f"\n" + "="*20 + " 发生致命错误! " + "="*20); print(f"!! 错误: {e}"); traceback.print_exc();
        input("\n程序已崩溃，按回车键退出。")
    finally: print("\n>> [系统] 程序已关闭。")