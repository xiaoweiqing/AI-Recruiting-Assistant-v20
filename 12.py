# ===================================================================================
#      AI 简历信息提取助理 v3.1 (Llama.cpp 纯文本稳定版)
# ===================================================================================
# v3.1 更新:
# - [终极修复] 放弃JSON输出，改用最稳定的纯文本“键: 值”格式提取，解决所有模型输出不稳定的问题。
# - [解析强化] 引入基于正则表达式的纯文本解析函数，鲁棒性远超JSON解析。
# - [提示词简化] 将提示词改为对模型最友好的“填空”模式，最大化成功率。
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
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models

import platform
import subprocess

load_dotenv()

# ==============================================================================
# ⬇⬇⬇ 0. 用户配置区 ⬇⬇⬇
# ==============================================================================
# --- 【【【 Llama.cpp 核心配置 (请务必填写!) 】】】 ---
LLAMACPP_PATH = "/home/weiyubin/llama.cpp/build/bin/llama-cli"
MODEL_PATH = "/home/weiyubin/models/gpt-oss-20b-Q4_K_M.gguf"
LLAMACPP_ARGS = "-n 2048 -c 4096 --temp 0.1 -t 8" # <-- 已移除 --instruct
# --------------------------------------------------------------------------

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
CANDIDATE_DB_ID = os.getenv("CANDIDATE_DB_ID")
CANDIDATE_PROFILE_HUB_DB_ID = os.getenv("CANDIDATE_PROFILE_HUB_DB_ID")

NOTION_PROPS = {
    "analysis_name": "候选人姓名", "analysis_date": "分析日期", "analysis_source": "源文件名",
    "analysis_phone": "联系电话", "analysis_email": "候选人邮箱",
    "profile_name": "姓名", "profile_relation_to_analysis": "关联分析报告",
    "profile_core_skills": "核心能力或技能", "profile_experience": "岗位经验",
    "profile_education": "学历", "profile_current_location": "目前所在地",
    "profile_expected_salary": "期望薪资", "profile_resume_hash": "简历内容哈希",
}

EMBEDDING_MODEL = None; QDRANT_CLIENT = None; IS_RAG_ENABLED = False
RECRUITMENT_COLLECTION_NAME = "ai_recruitment_assistant_embeddinggemma_v3"
WORKER_COUNT = 4

# ==============================================================================
# ⬇⬇⬇ AI 信息提取 (最终简化版 - 纯文本) ⬇⬇⬇
# ==============================================================================
EXTRACT_PROMPT_V3_CN = """### Instruction:
你是一个信息提取机器人。请从下面的简历文本中提取关键信息，并严格按照 "键: 值" 的格式，每项信息占一行。如果找不到某项信息，就留空。

姓名: 
联系电话: 
候选人邮箱: 
学历: 
岗位经验: 
目前所在地: 
期望薪资: 
核心能力或技能: 
一个简短的专业总结段落: 

---
待分析的简历文本：
{resume_text}

### Response:
姓名: 
"""

def parse_plain_text_extraction(text: str) -> dict:
    """
    从模型生成的纯文本中解析出键值对信息。
    """
    print("   -> [文本解析] 开始从以下文本中解析信息:")
    # 只打印前1000个字符以防日志过长
    print(text[:1000] + "..." if len(text) > 1000 else text)
    
    # 移除模型可能添加的前缀指令部分
    if "### Response:" in text:
        text = text.split("### Response:")[-1]

    extracted_data = {}
    patterns = {
        "name": r"姓名:\s*(.*)",
        "phone": r"联系电话:\s*(.*)",
        "email": r"候选人邮箱:\s*(.*)",
        "education": r"学历:\s*(.*)",
        "work_experience_summary": r"岗位经验:\s*(.*)",
        "current_location": r"目前所在地:\s*(.*)",
        "expected_salary": r"期望薪资:\s*(.*)",
        "core_skills": r"核心能力或技能:\s*(.*)",
        "full_summary": r"一个简短的专业总结段落:\s*([\s\S]*)", # 使用[\s\S]*来匹配包括换行在内的所有字符
    }
    
    temp_summary = ""
    lines = text.splitlines()
    summary_started = False

    for line in lines:
        if summary_started:
            # 如果总结已经开始，就把所有后续行都加进去，直到遇到下一个明确的键
            if not any(line.strip().startswith(key + ':') for key in patterns):
                temp_summary += "\n" + line
            else:
                summary_started = False # 遇到新键，总结结束
        
        for key, pattern in patterns.items():
            # 我们用 startswith 来做更简单的匹配
            if line.strip().startswith(key.replace('_', ' ') + ':'):
                 value = line.split(':', 1)[-1].strip()
                 if value:
                     extracted_data[key] = value
                 if key == 'full_summary':
                     summary_started = True
                     temp_summary = value # 初始化总结内容
                     
    if temp_summary:
        extracted_data['full_summary'] = temp_summary.strip()
    
    # 对提取出的电话和邮箱做一次清洗
    if 'phone' in extracted_data:
        phone_match = re.search(r'\d{11}', extracted_data['phone'])
        if phone_match: extracted_data['phone'] = phone_match.group(0)
    if 'email' in extracted_data:
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', extracted_data['email'])
        if email_match: extracted_data['email'] = email_match.group(0)

    print(f"   -> [文本解析] 解析结果: {extracted_data}")
    return extracted_data


# ==============================================================================
# ⬇⬇⬇ 系统设置 & 核心辅助函数 ⬇⬇⬇
# ==============================================================================
async def run_inference_with_llamacpp(prompt: str) -> str:
    """
    通过调用 llama.cpp 外部程序来异步执行 LLM 推理。
    """
    command = f'"{LLAMACPP_PATH}" -m "{MODEL_PATH}" {LLAMACPP_ARGS}'
    
    proc = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate(input=prompt.encode('utf-8'))

    if proc.returncode != 0:
        error_message = stderr.decode('utf-8', errors='ignore')
        print(f"   !! [Llama.cpp] 脚本执行出错。Stderr: {error_message[:500]}...")

    return stdout.decode('utf-8', errors='ignore')


def show_desktop_notification(title, message):
    if platform.system() == "Linux":
        try:
            print(f"\n🔔 [提醒] 准备发送通知: '{title}'")
            subprocess.Popen(['notify-send', title, message, '-a', 'AI简历助理', '-t', '8000'])
            print("✅ [提醒] 通知命令已成功发出。")
        except Exception as e:
            print(f"!! [提醒] 发送通知失败: {e}")

async def setup_qdrant_and_embedding():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED
    try:
        print(">> [知识库] 正在初始化 Ollama embedding 模型..."); EMBEDDING_MODEL = OllamaEmbeddings(model="embeddinggemma")
        vector_size = await asyncio.to_thread(EMBEDDING_MODEL.embed_query, "test"); print(f">> [知识库] 向量维度: {len(vector_size)}。")
        print(">> [知识库] 正在连接到 Qdrant..."); QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
        collections_response = await asyncio.to_thread(QDRANT_CLIENT.get_collections)
        if RECRUITMENT_COLLECTION_NAME not in [c.name for c in collections_response.collections]:
            print(f">> [知识库] 集合 '{RECRUITMENT_COLLECTION_NAME}' 不存在，正在创建..."); await asyncio.to_thread(QDRANT_CLIENT.recreate_collection, collection_name=RECRUITMENT_COLLECTION_NAME, vectors_config=models.VectorParams(size=len(vector_size), distance=models.Distance.COSINE))
        print("✅ [知识库] 成功连接！"); IS_RAG_ENABLED = True
    except Exception as e: print(f"❌ [知识库] 初始化失败: {e} (请确保Ollama服务正在为Embedding运行)"); IS_RAG_ENABLED = False

async def check_llamacpp_setup():
    print(f">> [AI大脑] 使用 Llama.cpp 引擎")
    if not os.path.isfile(LLAMACPP_PATH):
        print(f"❌ [AI大脑] 配置错误！找不到 Llama.cpp 程序。"); print(f"   -> 当前路径: {LLAMACPP_PATH}"); return False
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ [AI大脑] 配置错误！找不到模型文件。"); print(f"   -> 当前路径: {MODEL_PATH}"); return False
    print("✅ [AI大脑] Llama.cpp 配置检查通过，已准备就绪。")
    return True

def read_file_content(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf': import fitz; doc = fitz.open(file_path); return "".join(page.get_text() for page in doc), None
        elif ext == '.docx': import docx; return "\n".join([p.text for p in docx.Document(file_path).paragraphs]), None
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read(), None
        else: return None, "Unsupported file type"
    except Exception as e: return None, str(e)

def get_content_hash(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

# ==============================================================================
# ⬇⬇⬇ Notion 交互 (逻辑不变) ⬇⬇⬇
# ==============================================================================
def format_json_for_notion(data: dict) -> list:
    if not data: return []
    pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
    chunks = [pretty_json[i:i + 1900] for i in range(0, len(pretty_json), 1900)]
    blocks = [{"type": "code", "code": {"rich_text": [{"type": "text", "text": {"content": chunks[0]}}], "language": "json"}}] if chunks else []
    for chunk in chunks[1:]:
         blocks.append({"type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}})
    return blocks

async def save_to_notion_async(async_notion, report_data: dict, resume_text: str, filename: str):
    if not report_data: print("   !! [Notion同步] 报告数据为空，跳过。"); return None, None
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
        print(f"✅ [Notion] 已创建分析报告 '{name}' 并归档完整数据。")
    except Exception as e: print(f"   !! [Notion] 操作分析报告库失败: {e}"); traceback.print_exc()

    if analysis_page_id:
        try:
            def create_rich_text(value):
                return {"rich_text": [{"text": {"content": str(value)}}]} if value else None
            profile_props = {
                NOTION_PROPS["profile_name"]: {"title": [{"text": {"content": name}}]},
                NOTION_PROPS["profile_relation_to_analysis"]: {"relation": [{"id": analysis_page_id}]},
                NOTION_PROPS["profile_resume_hash"]: create_rich_text(get_content_hash(resume_text)),
                NOTION_PROPS["profile_core_skills"]: create_rich_text(report_data.get("core_skills")),
                NOTION_PROPS["profile_education"]: create_rich_text(report_data.get("education")),
                NOTION_PROPS["profile_current_location"]: create_rich_text(report_data.get("current_location")),
                NOTION_PROPS["profile_expected_salary"]: create_rich_text(report_data.get("expected_salary")),
                NOTION_PROPS["profile_experience"]: create_rich_text(report_data.get("work_experience_summary")),
            }
            new_profile_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_PROFILE_HUB_DB_ID}, properties={k: v for k, v in profile_props.items() if v})
            profile_page_id = new_profile_page.get('id')
            print(f"✅ [Notion] 已为 '{name}' 创建详细档案并自动填充属性。")
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
        print(f"   -> [向量化] '{name}' 的档案已向量化。")
    except Exception as e: print(f"   !! [向量化] 失败: {e}")

# ==============================================================================
# ⬇⬇⬇ 核心处理流程 ⬇⬇⬇
# ==============================================================================
async def process_single_resume_async(file_path, worker_name):
    filename = os.path.basename(file_path); loop = asyncio.get_running_loop();
    resume_text, error_msg = await loop.run_in_executor(None, read_file_content, file_path)
    if error_msg or not resume_text or len(resume_text.strip()) < 50: return "skipped_empty"
    
    prompt_content = EXTRACT_PROMPT_V3_CN.format(resume_text=resume_text[:8000])
    
    print(f"\n>> [{worker_name}] 正在从 '{filename}' 深度提取信息 (via Llama.cpp)..."); 
    print("   " + "="*20 + " AI提取中 " + "="*20); 
    
    full_response_content = await run_inference_with_llamacpp(prompt_content)
    
    print("\n   " + "="*20 + " 提取完毕 " + "="*22);
    
    # --- 【【【 核心修改：调用新的纯文本解析函数 】】】 ---
    extracted_data = parse_plain_text_extraction(full_response_content)
    
    if not extracted_data or not extracted_data.get("name"): raise Exception(f"AI未能从 '{filename}' 提取出有效数据或姓名。")
    
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
    if not resumes_to_process: print(f"\n❌ 在 '{RESUMES_DIR}' 文件夹中没有找到简历。"); return;
    print(f"\n找到 {len(resumes_to_process)} 份简历, 准备高速提取..."); queue = asyncio.Queue(); result_lists = ([], [], []);
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
# ⬇⬇⬇ 主程序入口 ⬇⬇⬇
# ==============================================================================
async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*10 + " AI 简历信息提取助理 v3.1 (Llama.cpp 纯文本稳定版) " + "="*10)
    print(">> 系统启动中..."); print("-" * 70)
    if not await check_llamacpp_setup():
        input("\n程序配置错误，按回车键退出。")
        sys.exit(1)
    
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