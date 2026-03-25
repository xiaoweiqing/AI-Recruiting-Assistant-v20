#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
#      AI 招聘助理 v34.0 - 纯本地高并发版 (Graduation Project)
# ===================================================================================
# 版本亮点 (基于 v20.0):
# - 【【【 核心改造：完全本地化 】】】
# - [AI引擎切换] 已将云端 Gemini API 彻底替换为本地模型，通过 LangChain 连接到本地服务 (如 LM Studio)。
# - [移除云依赖] 完全移除了所有 Google Search API 相关的功能和配置，程序不再需要任何外部网络API。
# - [流程简化] 由于移除了网络搜索，将原有的两步分析（预分析+深度分析）合并为一步，提高了效率。
# - [保留核心架构] 完整保留了 Qdrant 向量数据库、本地 Embedding 模型、高并发处理流水线和 Notion 集成。
# ===================================================================================

import time
# --- [MODIFIED] 移除 google.generativeai ---
# import google.generativeai as genai
import sys
import os
import re
import json
import notion_client
from datetime import datetime, timezone, timedelta
import shutil
from notion_client.errors import APIResponseError
import hashlib
# --- [MODIFIED] 移除 googleapiclient ---
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
import traceback
import asyncio

# --- [MODIFIED] 引入 LangChain 和 Qdrant, 本地模型 ---
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# ==============================================================================
# ⬇⬇⬇ 【【【 在这里加入代理清理代码 】】】 ⬇⬇⬇
# ==============================================================================
# --- 【关键修复】取消系统代理环境变量，确保本地连接正常 ---
for proxy_var in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if proxy_var in os.environ:
        print(f">> [Proxy Cleaner] 发现并移除了系统代理设置: {proxy_var}")
        del os.environ[proxy_var]
# ==============================================================================
# ⬆⬆⬆ 【【【 添加结束 】】】 ⬆⬆⬆
# ==============================================================================

# ==============================================================================
# ⬇⬇⬇ 0. 用户配置区 (所有密钥和ID集中于此) ⬇⬇⬇
# ==============================================================================
# 请替换为您自己的密钥和ID
# --- 核心API密钥 ---
# --- [MODIFIED] 移除了 Google API Key，现在无需任何云端密钥 ---
# API_KEY = "AIzaSyByySuSY3rz8-_T0ndvxm_qNTAMLZhp4h4"
NOTION_TOKEN = "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" # Notion Integration Token

# --- [MODIFIED] 移除了 Google Search 全部配置 ---
# ENABLE_GOOGLE_SEARCH = True
# GOOGLE_API_KEY = "..."
# GOOGLE_CSE_ID = "..."

# --- Notion Database IDs ---
JD_HUB_DATABASE_ID = "23a584b1cda380ed8d7bceb856c6dc0c"
CANDIDATE_DB_ID = "22a584b1cda3802dbb5dd8e1aba9a967"
CANDIDATE_PROFILE_HUB_DB_ID = "238584b1cda380888c78ee2ca403cd72"
TRAINING_HUB_DATABASE_ID = "231584b1cda380a1927be2ab6f22cf33"
# --- [MODIFIED] 移除了公司情报库 Company DB ID ---
# COMPANY_DB_ID = "..."

# --- Notion属性名配置 ---
NOTION_PROPS = {
    # 候选人分析库
    "analysis_name": "候选人姓名", "analysis_score": "匹配度评分", "analysis_reason": "评分理由",
    "analysis_date": "分析日期", "analysis_source": "源文件名", "analysis_phone": "联系电话",
    "analysis_email": "候选人邮箱", "analysis_priority": "优先级", "analysis_best_fit_position": "最匹配职位",
    "analysis_top_score": "最高匹配分", "analysis_update_flag": "简历已更新",
    # --- [MODIFIED] 移除了与公司情报库的关联属性 ---
    # "analysis_relation_to_company": "关联公司情报",

    # JD库
    "jd_hard_requirements": "硬性门槛", "jd_status": "状态", "jd_title": "职位名称",

    # 候选人信息库 (Profile Hub)
    "profile_name": "姓名", "profile_age": "年龄", "profile_gender": "性别", "profile_education": "学历",
    "profile_experience": "岗位经验", "profile_relation_to_analysis": "关联分析报告",
    "profile_employment_status": "离职状态", "profile_core_skills": "核心能力或技能",
    "profile_current_location": "目前所在地", "profile_expected_city": "期望城市",
    "profile_current_salary": "目前薪资", "profile_expected_salary": "期望薪资",
    "profile_availability": "到岗时间", "profile_copy_paste_area": "一键复制区",
    "profile_resume_hash": "简历内容哈希",
    # --- [MODIFIED] 移除了网页摘要和公司关联属性 ---
    # "profile_web_summary": "背景调查摘要",
    "profile_similar_candidates": "历史相似人才",
    # "profile_relation_to_company": "关联任职公司情报",

    # 训练中心库
    "training_task_title": "训练任务", "training_task_type": "任务类型", "training_input": "源数据 (Input)",
    "training_output": "理想输出 (Output)", "training_status": "标注状态", "training_relation_to_candidate": "源链接-候选人中心",
    
    # --- [MODIFIED] 移除了公司情报库的所有属性 ---
}

# --- 系统与模型配置 ---
JOB_SEPARATOR = "---JOB_SEPARATOR---"
ACTIVE_JD_DATA = {}

# --- [MODIFIED] 配置本地AI模型和Qdrant ---
llm = None  # 将用于存储本地LLM的连接
QDRANT_CLIENT = None
EMBEDDING_MODEL = None
IS_RAG_ENABLED = False
RECRUITMENT_COLLECTION_NAME = "ai_recruitment_assistant_v15_data"
WORKER_COUNT = 8

# ==============================================================================
# ⬇⬇⬇ 系统设置 & 核心辅助函数 (v34.0 纯本地版) ⬇⬇⬇
# ==============================================================================

def setup_qdrant_and_embedding():
    global QDRANT_CLIENT, EMBEDDING_MODEL, IS_RAG_ENABLED
    try:
        # --- [MODIFIED] ---
        print(">> [知识库] 正在加载本地 embedding 模型 (all-MiniLM-L6-v2)...")
        # 【【【 修改为新的模型名称 】】】
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        vector_size = EMBEDDING_MODEL.get_sentence_embedding_dimension()
        # ... 后面的代码保持不变 ...
        
        print(">> [知识库] 正在连接到中央数据库枢纽 (localhost:6333)...")
        QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)

        collections_response = QDRANT_CLIENT.get_collections()
        collection_names = [col.name for col in collections_response.collections]
        if RECRUITMENT_COLLECTION_NAME not in collection_names:
            print(f">> [知识库] 集合 '{RECRUITMENT_COLLECTION_NAME}' 不存在，正在创建...")
            QDRANT_CLIENT.recreate_collection(
                collection_name=RECRUITMENT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
        
        count_result = QDRANT_CLIENT.count(collection_name=RECRUITMENT_COLLECTION_NAME, exact=True)
        print(f"✅ [知识库] 成功连接！当前招聘库中有 {count_result.count} 条记录。")
        IS_RAG_ENABLED = True
        return True
    except Exception as e:
        print(f"❌ [知识库] 严重错误: 无法连接或设置 Qdrant: {e}")
        print("   -> 请确保您已通过 docker run 命令正确启动了 ai_database_hub 容器！")
        IS_RAG_ENABLED = False
        return False

# --- [NEW] 新增本地LLM设置函数 ---
def setup_local_llm():
    global llm
    try:
        print(f">> [AI] 正在连接到本地模型 API at [http://127.0.0.1:8087/v1]...")
        llm = ChatOpenAI(
            openai_api_base="http://127.0.0.1:8087/v1", 
            openai_api_key="na", 
            model_name="local", 
            temperature=0.1, 
            max_tokens=8192, # 根据您的模型调整
            request_timeout=600
        )
        # 增加一个简单的调用来立即测试连接是否成功
        llm.invoke("Hi")
        print(f"✅ [AI] 本地模型连接成功。")
        return True
    except Exception as e:
        print(f"❌ [AI] 严重错误: 连接本地模型失败: {e}")
        print("   -> 请确保您的本地AI服务 (如 LM Studio) 正在运行，并且地址和端口正确。")
        return False

# --- [MODIFIED] 移除了 configure_api() 函数 ---

async def get_page_content_async(async_notion_client, page_id):
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
        print(">> [JD加载器] 正在从Notion同步 'Active' 状态的职位描述 (JD)...")
        async_notion = notion_client.AsyncClient(auth=NOTION_TOKEN)
        jds = {}
        active_pages = await notion_client.helpers.async_collect_paginated_api(
            async_notion.databases.query,
            database_id=JD_HUB_DATABASE_ID,
            filter={"property": NOTION_PROPS["jd_status"], "select": {"equals": "Active"}}
        )
        if not active_pages:
            print(f"⚠️ [JD加载器] 未在Notion中找到任何状态为 'Active' 的JD。")
            return False
        print(f">> [深度抓取] 发现 {len(active_pages)} 个有效职位，开始并发抓取正文内容...")
        async def fetch_jd_content(page):
            page_id = page.get('id')
            props = page.get('properties', {})
            title_list = props.get(NOTION_PROPS["jd_title"], {}).get('title', [])
            title = title_list[0].get('plain_text', f"Unknown_{page_id}") if title_list else f"Unknown_{page_id}"
            hard_reqs_prop_text = "".join(t.get('plain_text', '') for t in props.get(NOTION_PROPS["jd_hard_requirements"], {}).get('rich_text', []))
            page_body_text = await get_page_content_async(async_notion, page_id)
            full_jd_text = f"职位核心职责与要求:\n{page_body_text}\n\n硬性门槛与关键资格:\n{hard_reqs_prop_text}".strip()
            if full_jd_text:
                print(f"  ✅ 成功抓取: {title}")
                return title, {"content": full_jd_text, "id": page_id}
            else:
                print(f"  ❌ 抓取失败或内容为空: {title}")
                return None, None
        tasks = [fetch_jd_content(page) for page in active_pages]
        results = await asyncio.gather(*tasks)
        jds = {title: data for title, data in results if title}
        ACTIVE_JD_DATA = jds
        if not jds:
            print(f"⚠️ [JD加载器] 所有 'Active' 的JD页面内容均为空，无法进行匹配。")
            return False
        print(f"✅ [JD加载器] 成功加载 {len(jds)} 个有效职位的完整JD。")
        return True
    except APIResponseError as e:
        print(f"❌ [JD加载器] Notion API错误: {e.body}")
        return False
    except Exception as e:
        print(f"❌ [JD加载器] 从Notion同步JD时发生严重错误: {e}\n{traceback.format_exc()}")
        return False

def clean_json_response(text):
    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match: json_str = match.group(1)
    else:
        start_index = text.find('{'); end_index = text.rfind('}')
        if start_index != -1 and end_index != -1: json_str = text[start_index : end_index + 1]
        else: return "{}"
    try:
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
    except Exception: return json_str

def read_file_content(file_path):
    try:
        if not os.path.exists(file_path): return None, "文件不存在"
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            import fitz
            with fitz.open(file_path) as doc: return "".join(page.get_text() for page in doc), None
        elif ext == '.docx':
            import docx
            return "\n".join([p.text for p in docx.Document(file_path).paragraphs]), None
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read(), None
        else: return None, f"不支持的文件类型: {ext}"
    except Exception as e: return None, str(e)

def get_content_hash(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

def split_text_for_notion(text, chunk_size=1999):
    if not text or not isinstance(text, str): return [{"type": "text", "text": {"content": ""}}]
    clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    text_chunks = [clean_text[i:i + chunk_size] for i in range(0, len(clean_text), chunk_size)]
    return [{"type": "text", "text": {"content": chunk}} for chunk in text_chunks]

# ==============================================================================
# ⬇⬇⬇ 超感知引擎 (仅向量搜索) [v34.0 纯本地版] ⬇⬇⬇
# ==============================================================================
# --- [MODIFIED] 移除了 perform_google_search_async 和 research_and_summarize_company_async ---

async def enrich_candidate_data_async(resume_text):
    print(">> [数据增强] 正在执行历史库相似人才搜索...")
    
    async def vector_search_task_async():
        if not IS_RAG_ENABLED: return "RAG功能未启用或无相似人才。"
        try:
            loop = asyncio.get_running_loop()
            query_vector = await loop.run_in_executor(None, EMBEDDING_MODEL.encode, resume_text)
            
            results = await loop.run_in_executor(None, lambda: QDRANT_CLIENT.search(
                collection_name=RECRUITMENT_COLLECTION_NAME,
                query_vector=query_vector.tolist(),
                limit=3,
                with_payload=True
            ))
            
            documents = [hit.payload.get('summary_text', 'No summary found') for hit in results]
            if documents:
                print(f"   ✅ [向量搜索] 成功找到 {len(documents)} 位相似的历史候选人。")
                return "\n\n---\n\n".join(documents)
            return "未找到相似人才。"
        except Exception as e:
            print(f"   !! [向量搜索] 检索相似候选人时出错: {e}")
            return "检索相似人才时出错。"

    similar_candidates_context = await vector_search_task_async()
    
    print("✅ [数据增强] 任务已完成。")
    return {
        "similar_candidates_context": similar_candidates_context
    }

# ==============================================================================
# ⬇⬇⬇ AI分析 & Prompting (v34.0 纯本地版) ⬇⬇⬇
# ==============================================================================
# --- [MODIFIED] 简化了Prompt，移除了所有与网络搜索和公司情报相关的内容 ---
LOCAL_AI_PROMPT = """
# 角色
你是一位顶级的AI技术招聘官，以严谨和注重细节著称。你的任务是整合所有信息，给出一份无可挑剔的分析报告。

# 任务
1.  **【质检】**: 首先，判断【输入文本】是否为一份有效的候选人简历。
2.  **【分析】**: 如果是有效简历，则严格对照【职位列表】中的每一项，并结合【历史相似人才参考】进行逐项分析，最终填充【输出JSON】。

# 核心规则
1.  **质检优先**: 如果【输入文本】不是简历，`is_resume` 设为 `false`，并留空其他所有字段。绝不允许分析非简历内容。
2.  **强制逐项分析**: 对于每个JD，必须将其要求分解，并对每一项进行独立的“是/否/不明确”评估。
3.  **忠于事实**: 所有判断必须基于简历原文。
4.  **完整提取档案**: 必须从简历中提取所有`profile_data`中要求的字段，找不到则填"N/A"。

# 输入材料
【输入1: 职位列表 (JDs from Notion)】
{all_jds_text}
---
【输入2: 候选人简历】
{resume_text}
---
【输入3: 历史相似人才参考 (from Vector DB)】
{similar_candidates_context}

# 输出格式 (严格遵循此JSON结构，不要输出任何额外文字)
{{
  "is_resume": true,
  "profile_data": {{
    "name": "...", "email": "...", "phone": "...", "age": 0, "gender": "N/A", "education": "...", "employment_status": "N/A",
    "work_experience_summary": "...", "core_competencies": "...", "current_location": "...", "expected_city": "N/A",
    "current_salary": "N/A", "expected_salary": "N/A", "availability": "N/A"
  }},
  "evaluation": {{
    "position_analysis": [
      {{
        "position_title": "...",
        "hard_req_breakdown": [
          {{"requirement": "...", "met": "是", "reasoning": "基于简历证据的判断"}}
        ],
        "strengths_summary": "...",
        "gaps_summary": "..."
      }}
    ]
  }},
  "overall_summary": {{
    "key_rationale": "综合所有信息，给出最终推荐或不推荐的核心理由。",
    "estimated_company_names": ["候选人主要任职过的公司名称列表"]
  }}
}}
"""

async def analyze_with_local_llm_async(resume_text, all_jds_text, enhancement_data, worker_name=""):
    if not resume_text or len(resume_text.strip()) < 50: return None
    
    prompt = LOCAL_AI_PROMPT.format(
        all_jds_text=all_jds_text, 
        resume_text=resume_text,
        similar_candidates_context=enhancement_data.get('similar_candidates_context', '无历史相似人才')
    )
    for attempt in range(3):
        try:
            print(f"\n[{worker_name}] [本地AI分析中...第 {attempt + 1}/3 次尝试... AI正在流式生成响应...]\n", end="", flush=True)
            
            # --- [MODIFIED] 使用本地 llm.astream 进行流式调用 ---
            response_stream = llm.astream(prompt)
            
            full_response_text = ""
            async for chunk in response_stream:
                content = chunk.content
                print(content, end="", flush=True)
                full_response_text += content
            
            print(f"\n[{worker_name}] [AI分析完成！]")
            if not full_response_text.strip():
                raise ValueError("本地模型返回了空内容")
            return full_response_text
            
        except Exception as e:
            print(f"\n[{worker_name}] 错误: {e}")
            if "DEADLINE_EXCEEDED" in str(e) or "Timeout" in str(e): 
                print("   -> 超时错误，模型可能在处理大型简历，请耐心等待重试...")
            if attempt < 2: await asyncio.sleep(8)
    return None

# ==============================================================================
# ⬇⬇⬇ 报告格式化 & Notion交互 (v34.0 纯本地版) ⬇⬇⬇
# ==============================================================================
def sanitize_parsed_data(parsed_data):
    if not isinstance(parsed_data, dict): return {}
    profile = parsed_data.get('profile_data', {}); profile['name'] = profile.get('name') or "未知候选人"
    profile['email'] = profile.get('email') or None; profile['phone'] = profile.get('phone') or None
    parsed_data['profile_data'] = profile
    evaluation = parsed_data.get('evaluation', {})
    if 'position_analysis' not in evaluation: evaluation['position_analysis'] = []
    parsed_data['evaluation'] = evaluation
    if 'overall_summary' not in parsed_data: parsed_data['overall_summary'] = {"key_rationale": "AI未能生成综合评估。", "estimated_company_names": []}
    return parsed_data

def calculate_and_format_report_v15(parsed_data, filename="N/A", resume_text=""):
    sanitized_data = sanitize_parsed_data(parsed_data)
    profile_data = sanitized_data.get("profile_data", {})
    analysis_list = sanitized_data.get("evaluation", {}).get("position_analysis", [])
    for analysis in analysis_list:
        breakdown = analysis.get("hard_req_breakdown", []); score_map = {"是": 1.0, "不明确": 0.5, "否": 0.0}
        met_count = sum(1 for item in breakdown if item.get("met") == "是")
        score_sum = sum(score_map.get(item.get("met", "否"), 0.0) for item in breakdown)
        analysis['score'] = round(5 + 5 * (score_sum / len(breakdown)), 1) if breakdown else 0.0
        analysis['match_stats'] = f"{len(breakdown)}条满足{met_count}条"
    sorted_list = sorted(analysis_list, key=lambda x: x.get('score', 0.0), reverse=True)
    top_score = sorted_list[0]['score'] if sorted_list else 0.0
    best_fit = sorted_list[0]['position_title'] if sorted_list else "无匹配职位"
    report_text = f"## AI 智能匹配报告\n\n**候选人**: {profile_data.get('name', 'N/A')}\n**最佳匹配职位**: {best_fit}\n**最高匹配分**: {top_score:.1f}\n**核心推荐理由**: {sanitized_data.get('overall_summary', {}).get('key_rationale', 'N/A')}\n\n---\n\n### 各职位详细分析\n"
    for item in sorted_list:
        report_text += f"\n#### **{item.get('position_title')}** - 评分: {item.get('score', 0.0):.1f} ({item.get('match_stats', '')})\n- **优势总结**: {item.get('strengths_summary', 'N/A')}\n- **差距总结**: {item.get('gaps_summary', 'N/A')}\n"
    return {"filename": filename, "name": profile_data.get('name'), "phone": profile_data.get('phone'), "email": profile_data.get('email'), "top_score": top_score, "best_fit": best_fit, "reason": sanitized_data.get('overall_summary', {}).get('key_rationale', 'N/A'), "full_report_text": report_text, "full_report_data": sanitized_data}

async def check_candidate_existence_async(async_notion, candidate_name, candidate_email):
    # (此函数无需修改)
    if not CANDIDATE_PROFILE_HUB_DB_ID: return None
    db_filter = {"or": []}
    if candidate_email and candidate_email.lower() != 'n/a':
        db_filter["or"].append({"property": NOTION_PROPS["profile_name"].replace("姓名", "候选人邮箱"), "email": {"equals": candidate_email}})
    if candidate_name and candidate_name not in ['N/A', '未知候选人']:
        db_filter["or"].append({"property": NOTION_PROPS["profile_name"], "title": {"equals": candidate_name}})
    if not db_filter["or"]: return None
    try:
        response = await async_notion.databases.query(database_id=CANDIDATE_PROFILE_HUB_DB_ID, filter=db_filter, page_size=1)
        if response and response['results']:
            page = response['results'][0]; page_id = page['id']
            hash_prop = page['properties'].get(NOTION_PROPS["profile_resume_hash"], {})
            old_hash = hash_prop.get('rich_text', [{}])[0].get('plain_text', '') if hash_prop.get('rich_text') else ''
            analysis_relation = page['properties'].get(NOTION_PROPS["profile_relation_to_analysis"], {}).get('relation', [])
            analysis_page_id = analysis_relation[0]['id'] if analysis_relation else None
            print(f"   -> [去重检查] 发现已存在候选人 '{candidate_name}' 的档案。")
            return {"profile_page_id": page_id, "analysis_page_id": analysis_page_id, "old_hash": old_hash}
    except Exception as e: 
        print(f"   !! [去重检查] 查询Notion时发生错误: {e}")
    return None

async def save_or_update_analysis_report_async(async_notion, report_data, resume_text, existing_page_id=None):
    # (此函数无需修改)
    beijing_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))); score = float(report_data.get('top_score', 0))
    if score >= 9: tag = "🌟 S级 (必须拿下)"
    elif score >= 8: tag = "🔥 A级 (重点跟进)"
    elif score >= 7: tag = "✅ B级 (符合预期)"
    else: tag = "🤔 C级 (待定观察)"
    action = "更新" if existing_page_id else "创建"
    props = {
        NOTION_PROPS["analysis_name"]: {"title": [{"text": {"content": report_data.get('name', '解析失败')}}]},
        NOTION_PROPS["analysis_phone"]: {"phone_number": report_data.get('phone') if report_data.get('phone') and report_data.get('phone') != 'N/A' else None},
        NOTION_PROPS["analysis_email"]: {"email": report_data.get('email') if report_data.get('email') and report_data.get('email') != 'N/A' else None},
        NOTION_PROPS["analysis_date"]: {"date": {"start": beijing_time.isoformat()}},
        NOTION_PROPS["analysis_source"]: {"rich_text": [{"text": {"content": report_data.get('filename', 'N/A')}}]},
        NOTION_PROPS["analysis_best_fit_position"]: {"rich_text": [{"text": {"content": report_data.get('best_fit', 'N/A')}}]},
        NOTION_PROPS["analysis_reason"]: {"rich_text": split_text_for_notion(report_data.get('reason', '无'))},
        NOTION_PROPS["analysis_top_score"]: {"number": score}, NOTION_PROPS["analysis_score"]: {"number": score},
        NOTION_PROPS["analysis_priority"]: {"select": {"name": tag}}
    }
    if action == "更新": props[NOTION_PROPS["analysis_update_flag"]] = {"checkbox": True}
    final_props = {k: v for k, v in props.items() if v is not None}
    print(f"\n>> [Notion同步] 正在{action} '{report_data.get('name', 'N/A')}' 的分析报告...")
    try:
        if existing_page_id:
            await async_notion.pages.update(page_id=existing_page_id, properties=final_props)
            old_blocks_resp = await async_notion.blocks.children.list(block_id=existing_page_id)
            for block in old_blocks_resp['results']: 
                try: await async_notion.blocks.delete(block_id=block['id'])
                except APIResponseError: pass
            page_id_to_return = existing_page_id
        else:
            new_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_DB_ID}, icon={"type": "emoji", "emoji": "🎯"}, properties=final_props)
            page_id_to_return = new_page.get('id')
        full_page_content_text = report_data['full_report_text'] + "\n\n---\n\n## 原始简历文本\n" + resume_text
        content_chunks = [full_page_content_text[i:i + 2000] for i in range(0, len(full_page_content_text), 2000)]
        children_to_append = [{"type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": chunk}}]}} for chunk in content_chunks]
        await async_notion.blocks.children.append(block_id=page_id_to_return, children=children_to_append[:100])
        print(f"✅ [Notion同步] 成功{action}分析报告！")
        return page_id_to_return
    except Exception as e: print(f"❌ 错误: {action}分析报告到Notion失败: {e}"); return None

async def save_or_update_candidate_profile_async(async_notion, report_data, analysis_page_id, enhancement_data, resume_text, existing_page_id=None):
    # --- [MODIFIED] 移除了 web_summary 的更新 ---
    full_report_data = report_data.get("full_report_data", {}); profile_data = full_report_data.get("profile_data", {})
    action = "更新" if existing_page_id else "创建"; print(f">> [档案{action}] 启动候选人档案流程...")
    def format_profile_copy_paste(data):
        lines = []
        if data.get('name'): lines.append(f"姓名: {data['name']}")
        if data.get('phone'): lines.append(f"电话: {data['phone']}")
        if data.get('email'): lines.append(f"邮箱: {data['email']}")
        if data.get('education'): lines.append(f"学历: {data['education']}")
        if data.get('work_experience_summary'): lines.append(f"经验: {data['work_experience_summary']}")
        return "\n".join(lines)
    props = {
        NOTION_PROPS["profile_name"]: {"title": [{"text": {"content": profile_data.get('name', 'N/A')}}]},
        NOTION_PROPS.get("profile_email"): {"email": profile_data.get('email') if profile_data.get('email') and profile_data.get('email') != 'N/A' else None},
        NOTION_PROPS["profile_age"]: {"number": int(profile_data.get('age', 0)) if str(profile_data.get('age', 0)).isdigit() else None},
        NOTION_PROPS["profile_gender"]: {"select": {"name": profile_data.get('gender')}} if profile_data.get('gender') not in ['N/A', '', None] else None,
        NOTION_PROPS["profile_education"]: {"rich_text": split_text_for_notion(profile_data.get('education', 'N/A'))},
        NOTION_PROPS["profile_employment_status"]: {"select": {"name": profile_data.get('employment_status')}} if profile_data.get('employment_status') not in ['N/A', '', None] else None,
        NOTION_PROPS["profile_experience"]: {"rich_text": split_text_for_notion(profile_data.get('work_experience_summary', 'N/A'))},
        NOTION_PROPS["profile_core_skills"]: {"rich_text": split_text_for_notion(profile_data.get('core_competencies', 'N/A'))},
        NOTION_PROPS["profile_current_location"]: {"rich_text": [{"text": {"content": profile_data.get('current_location', 'N/A')}}]},
        NOTION_PROPS["profile_expected_city"]: {"rich_text": [{"text": {"content": profile_data.get('expected_city', 'N/A')}}]},
        NOTION_PROPS["profile_availability"]: {"rich_text": [{"text": {"content": profile_data.get('availability', 'N/A')}}]},
        NOTION_PROPS["profile_resume_hash"]: {"rich_text": [{"text": {"content": get_content_hash(resume_text)}}]},
        NOTION_PROPS["profile_copy_paste_area"]: {"rich_text": split_text_for_notion(format_profile_copy_paste(profile_data))},
        NOTION_PROPS["profile_similar_candidates"]: {"rich_text": split_text_for_notion(str(enhancement_data.get('similar_candidates_context', 'N/A')))},
    }
    if analysis_page_id: props[NOTION_PROPS["profile_relation_to_analysis"]] = {"relation": [{"id": analysis_page_id}]}
    final_props = {k: v for k, v in props.items() if k and v is not None}
    try:
        if existing_page_id:
            await async_notion.pages.update(page_id=existing_page_id, properties=final_props)
            page_id_for_vector = existing_page_id
        else:
            new_profile_page = await async_notion.pages.create(parent={"database_id": CANDIDATE_PROFILE_HUB_DB_ID}, properties=final_props)
            page_id_for_vector = new_profile_page.get('id')
        print(f"✅ [Notion同步] 成功将档案{action}到 [候选人信息库]！")
        asyncio.create_task(vectorize_and_store_profile_async(page_id_for_vector, profile_data, report_data))
        return page_id_for_vector
    except Exception as e: print(f"❌ 错误: {action}到'候选人信息库'失败: {e}"); return None

# --- [MODIFIED] 移除了 sync_company_to_notion_and_vectorize_async 函数 ---

# ==============================================================================
# ⬇⬇⬇ 向量化 & 训练中心 (v34.0 纯本地版) ⬇⬇⬇
# ==============================================================================
async def vectorize_and_store_profile_async(page_id, profile_data, full_report_data):
    # (此函数无需修改)
    if not IS_RAG_ENABLED or not profile_data: return
    candidate_name = profile_data.get('name', '未知候选人')
    print(f"   -> [后台任务] 正在向量化候选人 '{candidate_name}' 的综合档案...")
    try:
        factual_summary = f"姓名: {profile_data.get('name')}, 学历: {profile_data.get('education')}, 经验: {profile_data.get('work_experience_summary')}, 核心能力: {profile_data.get('core_competencies')}"
        analysis_summary = f"AI首要推荐: {full_report_data['best_fit']}, 理由: {full_report_data['reason']}"
        comprehensive_text = f"{factual_summary}\n\n--- AI分析摘要 ---\n{analysis_summary}"
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, EMBEDDING_MODEL.encode, comprehensive_text)
        payload = { "summary_text": comprehensive_text, "source_title": f"候选人综合档案: {candidate_name}" }
        await loop.run_in_executor(None, lambda: QDRANT_CLIENT.upsert(
            collection_name=RECRUITMENT_COLLECTION_NAME,
            points=[models.PointStruct(id=str(page_id), vector=vector.tolist(), payload=payload)],
            wait=True
        ))
        print(f"   ✅ [后台任务] 成功将 '{candidate_name}' 的档案向量化！")
    except Exception as e:
        print(f"   !! [后台任务] 向量化候选人档案时出错: {e}")

# --- [MODIFIED] 移除了 vectorize_company_data_async 函数 ---

async def write_to_training_hub_async(async_notion, resume_text, parsed_analysis_data, analysis_page_id):
    # --- [MODIFIED] 移除了 company_research_results 参数 ---
    if not TRAINING_HUB_DATABASE_ID: return
    print("   -> [后台任务] 正在写入训练中心...")
    try:
        output_str = json.dumps(parsed_analysis_data, ensure_ascii=False, indent=2)
        title = f"【简历分析】{parsed_analysis_data.get('profile_data', {}).get('name', '未知')}"
        props = {
            NOTION_PROPS["training_task_title"]: {"title": [{"text": {"content": title}}]},
            NOTION_PROPS["training_task_type"]: {"select": {"name": "简历分析-本地版"}},
            NOTION_PROPS["training_input"]: {"rich_text": split_text_for_notion(resume_text)},
            NOTION_PROPS["training_output"]: {"rich_text": split_text_for_notion(output_str)},
            NOTION_PROPS["training_status"]: {"select": {"name": "待审核"}},
        }
        if analysis_page_id:
            props[NOTION_PROPS["training_relation_to_candidate"]] = {"relation": [{"id": analysis_page_id}]}
        await async_notion.pages.create(parent={"database_id": TRAINING_HUB_DATABASE_ID}, properties=props)
        print("   ✅ [后台任务] 成功写入训练中心！")
    except Exception as e:
        print(f"   !! [后台任务] 写入训练中心时发生错误: {e}")

# ==============================================================================
# ⬇⬇⬇ 主执行模式 (v34.0 纯本地高并发流水线) ⬇⬇⬇
# ==============================================================================

async def process_single_resume_async(file_path, all_jds_text, worker_name):
    """
    这是单个简历的完整异步处理逻辑，由一个worker调用。
    --- [MODIFIED] 简化了整个流程，移除了两步分析 ---
    """
    filename = os.path.basename(file_path)
    async_notion = notion_client.AsyncClient(auth=NOTION_TOKEN)
    loop = asyncio.get_running_loop()
    
    # 步骤 1: 读取文件
    print(f">> [{worker_name}] [流程 1/4] 读取文件内容...")
    resume_text, error_msg = await loop.run_in_executor(None, read_file_content, file_path)
    if error_msg: raise Exception(f"读取文件失败: {error_msg}")

    # 步骤 2: 智能去重检查
    print(f">> [{worker_name}] [流程 2/4] 智能去重与数据增强...")
    # 为了去重，我们需要先做一次快速的AI分析来提取姓名和邮箱
    temp_analysis_data = await analyze_with_local_llm_async(resume_text, "仅提取姓名和邮箱", {}, worker_name)
    parsed_temp_data = await loop.run_in_executor(None, json.loads, clean_json_response(temp_analysis_data))
    profile_data = parsed_temp_data.get('profile_data', {})
    candidate_name = profile_data.get('name', '未知候选人'); candidate_email = profile_data.get('email')

    existing_candidate_info = await check_candidate_existence_async(async_notion, candidate_name, candidate_email)
    if existing_candidate_info and existing_candidate_info.get("old_hash") == get_content_hash(resume_text):
        print(f"   -> [{worker_name}] [跳过] 候选人 '{candidate_name}' 已存在且简历无变化。")
        return "skipped_no_change"
    
    # 执行数据增强（目前只有向量搜索）
    enhancement_data = await enrich_candidate_data_async(resume_text)

    # 步骤 3: 进行最终的AI分析
    print(f"\n>> [{worker_name}] [流程 3/4] 进行AI深度分析...")
    final_analysis_json_text = await analyze_with_local_llm_async(resume_text, all_jds_text, enhancement_data, worker_name)
    if not final_analysis_json_text: raise Exception("AI最终分析返回空内容")
    
    final_parsed_data = await loop.run_in_executor(None, json.loads, clean_json_response(final_analysis_json_text))
    if not final_parsed_data.get("is_resume"):
        print(f">> [{worker_name}] AI判断 '{filename}' 不是有效简历，将跳过。")
        return "skipped_not_resume"
    
    # 步骤 4: 解析最终报告并同步到Notion
    print(f"\n>> [{worker_name}] [流程 4/4] 解析报告并同步到Notion...")
    report_data = await loop.run_in_executor(None, calculate_and_format_report_v15, final_parsed_data, filename, resume_text)
    
    existing_analysis_id = existing_candidate_info.get("analysis_page_id") if existing_candidate_info else None
    analysis_page_id = await save_or_update_analysis_report_async(async_notion, report_data, resume_text, existing_analysis_id)
    if not analysis_page_id: raise Exception("未能创建或更新Notion分析报告页面")
    
    existing_profile_id = existing_candidate_info.get("profile_page_id") if existing_candidate_info else None
    profile_page_id = await save_or_update_candidate_profile_async(async_notion, report_data, analysis_page_id, enhancement_data, resume_text, existing_profile_id)
    if not profile_page_id: print(f"⚠️ [{worker_name}] 警告：成功同步分析报告，但档案同步失败。")

    # --- [MODIFIED] 移除了公司情报同步的相关逻辑 ---

    # 启动后台任务，写入训练中心
    asyncio.create_task(write_to_training_hub_async(async_notion, resume_text, final_parsed_data, analysis_page_id))
    
    return {"status": "success", "report": report_data}


async def resume_worker(name, queue, all_jds_text, result_lists):
    # (此函数无需修改)
    processed_summaries, failed_files, skipped_files = result_lists
    while True:
        try:
            file_path = await queue.get()
            filename = os.path.basename(file_path)
            print(f"[{name}] 开始处理: {filename}")
            PROCESSED_DIR, FAILED_DIR, SKIPPED_DIR = "processed_resumes_batch_match", "processed_failed", "processed_skipped_duplicates"
            try:
                result = await process_single_resume_async(file_path, all_jds_text, name)
                if isinstance(result, str) and result.startswith("skipped"):
                    if result == "skipped_not_resume":
                        skipped_files.append(f"{filename} (非简历)")
                        shutil.move(file_path, os.path.join(SKIPPED_DIR, filename))
                    elif result == "skipped_no_change":
                        skipped_files.append(f"{filename} (简历未变)")
                        shutil.move(file_path, os.path.join(SKIPPED_DIR, filename))
                elif isinstance(result, dict) and result.get("status") == "success":
                    report_data = result['report']
                    processed_summaries.append({
                        "name": report_data['name'], "score": report_data['top_score'], "position": report_data['best_fit']
                    })
                    print(f"\n✅ [{name}] 候选人 '{report_data['name']}' ({filename}) 分析与同步完成。")
                    shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
            except Exception as e:
                print(f"!! [{name}] 处理 {filename} 失败: {e}\n{traceback.format_exc()}");
                failed_files.append(f"{filename} (错误: {str(e)[:50]}...)")
                shutil.move(file_path, os.path.join(FAILED_DIR, filename))
            finally:
                queue.task_done()
        except asyncio.CancelledError:
            break

async def batch_mode_high_concurrency():
    # (此函数无需修改)
    print("\n" + "="*14 + " 批量智能匹配模式 (v34.0 纯本地版) " + "="*14)
    RESUMES_DIR, PROCESSED_DIR = "resumes_to_process", "processed_resumes_batch_match"
    FAILED_DIR, SKIPPED_DIR = "processed_failed", "processed_skipped_duplicates"
    for d in [RESUMES_DIR, PROCESSED_DIR, FAILED_DIR, SKIPPED_DIR]: os.makedirs(d, exist_ok=True)
    resumes_to_process = [f for f in os.listdir(RESUMES_DIR) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    if not resumes_to_process:
        print(f"\n在 '{RESUMES_DIR}' 文件夹中没有找到简历。"); return
    print(f"\n找到 {len(resumes_to_process)} 份简历, 将与 {len(ACTIVE_JD_DATA)} 个有效职位进行匹配...")
    all_jds_text = JOB_SEPARATOR.join([f"职位名称: {title}\n\n职位详情:\n{data['content']}" for title, data in ACTIVE_JD_DATA.items()])
    queue = asyncio.Queue()
    processed_summaries, failed_files, skipped_files = [], [], []
    result_lists = (processed_summaries, failed_files, skipped_files)
    for filename in resumes_to_process:
        await queue.put(os.path.join(RESUMES_DIR, filename))
    worker_tasks = []
    for i in range(WORKER_COUNT):
        task = asyncio.create_task(resume_worker(f"Worker-{i+1}", queue, all_jds_text, result_lists))
        worker_tasks.append(task)
    await queue.join()
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    print("\n" + "="*70 + "\n✅ 所有简历批量智能匹配完毕！")
    print("="*28 + " 最终总结报告 " + "="*28)
    print(f"\n成功处理: {len(processed_summaries)} 份")
    if processed_summaries:
        print(f"| {'候选人姓名':<20} | {'最高分':<8} | {'最匹配职位':<30} |")
        print(f"|{'-'*22}|{'-'*10}|{'-'*32}|")
        sorted_summaries = sorted(processed_summaries, key=lambda x: x['score'], reverse=True)
        for summary in sorted_summaries:
            name_str = str(summary['name']); pos_str = str(summary['position'])
            name = (name_str[:17] + '...') if len(name_str) > 20 else name_str
            pos = (pos_str[:27] + '...') if len(pos_str) > 30 else pos_str
            print(f"| {name:<20} | {summary['score']:<8.1f} | {pos:<30} |")
    print(f"\n跳过处理: {len(skipped_files)} 份"); [print(f"  - {f}") for f in skipped_files]
    print(f"\n处理失败: {len(failed_files)} 份"); [print(f"  - {f}") for f in failed_files]
    print("="*70)

# ==============================================================================
# ⬇⬇⬇ 人才激活模式 & 程序入口 (v34.0 纯本地版) ⬇⬇⬇
# ==============================================================================
def talent_activation_mode():
    # (此函数无需修改)
    if not IS_RAG_ENABLED: print("\n错误: 无法执行人才激活，RAG功能未启用。"); return
    if not ACTIVE_JD_DATA: print("\n错误: 未加载到任何有效JD，无法进行人才激活。"); return
    print("\n" + "="*23 + " 人才激活模式 " + "="*23)
    jd_titles = list(ACTIVE_JD_DATA.keys())
    for i, title in enumerate(jd_titles): print(f"  {i + 1}. {title}")
    while True:
        try:
            choice_idx = int(input(f"\n请选择一份用于激活人才的JD (1-{len(jd_titles)}): ").strip()) - 1
            if 0 <= choice_idx < len(jd_titles):
                selected_title, selected_jd_content = jd_titles[choice_idx], ACTIVE_JD_DATA[jd_titles[choice_idx]]['content']
                break
            else: print("无效选择。")
        except (ValueError, IndexError): print("无效输入。")
    top_n = int(input("希望看到最匹配的前几位候选人？ (默认5): ").strip() or 5)
    print(f"\n>> [人才激活] 正在用【{selected_title}】在历史库中寻找前 {top_n} 位候选人...")
    try:
        query_vector = EMBEDDING_MODEL.encode(selected_jd_content).tolist()
        results = QDRANT_CLIENT.search(collection_name=RECRUITMENT_COLLECTION_NAME, query_vector=query_vector, limit=top_n, with_payload=True)
        if not results:
            print("\n>> [人才激活] 未在历史库中找到任何匹配的候选人档案。"); return
        print("\n" + "="*25 + " 人才激活推荐榜单 " + "="*25)
        for i, hit in enumerate(results):
            title = hit.payload.get('source_title', f"候选人 {i+1}")
            similarity_score = hit.score * 100
            doc = hit.payload.get('summary_text', '摘要不可用')
            print(f"\n--- Top {i+1}: {title} (匹配度: {similarity_score:.2f}%) ---")
            print(doc)
        print("\n" + "="*68)
    except Exception as e: print(f"\n!! [人才激活] 查询时发生错误: {e}")

async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*10 + " AI 招聘助理 v34.0 - 纯本地高并发版 " + "="*10)
    print(">> 系统启动: 正在初始化本地模型和数据库连接...")
    print("-" * 70)
    
    # --- [MODIFIED] 更新了启动检查流程 ---
    if not (setup_local_llm() and setup_qdrant_and_embedding() and await load_active_jd_from_notion_async()):
        input("核心系统初始化失败，请检查错误信息后按回车键退出。")
        sys.exit(1)

    while True:
        print("\n" + "="*28 + " 主菜单 " + "="*28)
        print("  1. [批量智配]   - (v34.0 纯本地版) 分析简历")
        print("  2. [人才激活]   - (RAG) 用JD在历史库中激活候选人")
        print("\n  Q. 退出程序")
        print("="*65)
        choice = input("请输入您的选择 (1, 2, 或 Q): ").strip().upper()
        if choice == '1':
            await batch_mode_high_concurrency()
        elif choice == '2':
            talent_activation_mode()
        elif choice == 'Q':
            print("\n程序已退出。")
            break
        else:
            print("\n无效输入，请重新选择。")
        input("\n当前模式任务已完成，按回车键返回主菜单...")
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n检测到用户中断，正在关闭...")
    finally:
        print(">> 所有任务完成，程序安全退出。")