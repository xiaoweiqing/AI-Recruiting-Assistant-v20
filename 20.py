import os
import json
import requests
from tqdm import tqdm
import time

# --- 配置信息 ---
# 请将您的凭证粘贴到此处
# 警告：直接在代码中写入密钥存在安全风险。更安全的方式是使用环境变量。
# For security, it's better to use environment variables instead of hardcoding keys.
NOTION_TOKEN = "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"
DATABASE_ID = "231584b1cda380a1927be2ab6f22cf33"  # 这是您提供的训练中心ID

# --- Notion API 配置 ---
NOTION_API_URL = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

# --- 微调配置 ---
# 您可以根据不同的任务类型自定义指令。
# 这里我们使用一个通用的指令，因为大部分任务是“简历分析”。
INSTRUCTION_PROMPT = "请从以下非结构化文本中，精确地提取出要求的信息。"
OUTPUT_FILE = "finetune_data.jsonl"


def get_plain_text_from_property(property_data):
    """
    从 Notion API 返回的 property 对象中安全地提取纯文本。
    """
    if not property_data:
        return ""

    # 兼容处理 title, rich_text 等多种类型
    if property_data.get("type") == "title":
        text_list = property_data.get("title", [])
    elif property_data.get("type") == "rich_text":
        text_list = property_data.get("rich_text", [])
    else:
        # 如果有其他类型的属性，可能需要在这里添加处理逻辑
        return ""

    return "".join([item.get("plain_text", "") for item in text_list]).strip()


def fetch_and_process_notion_data():
    """
    从 Notion 数据库获取、清洗并处理数据，增加了超时和重试机制。
    """
    all_formatted_data = []
    has_more = True
    next_cursor = None

    # --- 新增：重试逻辑 ---
    max_retries = 5

    print(f"🚀 开始从 Notion 数据库 (ID: ...{DATABASE_ID[-6:]}) 拉取数据...")

    with tqdm(desc="正在拉取页面", unit="页") as pbar:
        while has_more:
            payload = {}
            if next_cursor:
                payload["start_cursor"] = next_cursor

            # --- 新增：为每个请求循环重试 ---
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        NOTION_API_URL,
                        headers=HEADERS,
                        json=payload,
                        timeout=60,  # --- 修改：超时时间延长到 60 秒 ---
                    )
                    response.raise_for_status()
                    data = response.json()

                    # 如果成功，就跳出重试循环
                    break

                except requests.exceptions.RequestException as e:
                    print(f"\n❌ 第 {attempt + 1}/{max_retries} 次请求失败: {e}")
                    if attempt < max_retries - 1:
                        print(f"将在 10 秒后重试...")
                        time.sleep(10)
                    else:
                        print("\n❌ 达到最大重试次数，脚本终止。")
                        print("请重点检查您的代理/VPN连接和网络环境。")
                        return None  # 重试全部失败，则返回 None

            # ... (下面的数据处理逻辑保持不变) ...

            results = data.get("results", [])

            for page in results:
                properties = page.get("properties", {})
                source_data_prop = properties.get("源数据 (Input)")
                ideal_output_prop = properties.get("理想输出 (Output)")
                input_text = get_plain_text_from_property(source_data_prop)
                output_text = get_plain_text_from_property(ideal_output_prop)

                if input_text and output_text:
                    formatted_entry = {
                        "instruction": INSTRUCTION_PROMPT,
                        "input": input_text,
                        "output": output_text,
                    }
                    all_formatted_data.append(formatted_entry)

            has_more = data.get("has_more", False)
            next_cursor = data.get("next_cursor")
            pbar.update(1)

    print(
        f"\n✅ 数据拉取完成！共找到 {len(all_formatted_data)} 条高质量、有效的训练数据。"
    )
    return all_formatted_data


def save_to_jsonl(data_list, filename):
    """
    将处理好的数据列表保存为 JSONL 文件。
    """
    if not data_list:
        print("⚠️ 没有可保存的数据。")
        return

    try:
        with open(filename, "w", encoding="utf-8") as f:
            for entry in data_list:
                json_record = json.dumps(entry, ensure_ascii=False)
                f.write(json_record + "\n")
        print(f"🎉 数据成功保存到文件: {filename}")
        print("现在，您可以将这个文件用于您的大模型微调流程了！")
    except IOError as e:
        print(f"\n❌ 文件写入失败: {e}")


if __name__ == "__main__":
    processed_data = fetch_and_process_notion_data()
    if processed_data:
        save_to_jsonl(processed_data, OUTPUT_FILE)
