import json

# --- 请在这里配置您的文件名 ---
# 您的原始数据文件名
input_filename = "finetune_data.jsonl"
# 我们要创建的、格式正确的新文件名
output_filename = "finetune_data_corrected.jsonl"
# -----------------------------

print(f"开始转换文件: {input_filename}")

# 读取原始的、包含整个列表的 JSON 文件
try:
    with open(input_filename, "r", encoding="utf-8") as f:
        # 将整个文件内容加载为一个 Python 列表
        data = json.load(f)
except json.JSONDecodeError:
    print("\n错误：文件格式似乎不是一个完整的 JSON 数组。")
    print(
        "请确认您的文件是以 '[' 开头，以 ']' 结尾，并且每个JSON对象之间用逗号 ',' 分隔。"
    )
    print("如果您的文件已经是 JSONL 格式（每行一个JSON），则无需转换。")
    exit()


# 写入新的、格式正确的 JSONL 文件
count = 0
with open(output_filename, "w", encoding="utf-8") as f_out:
    # 遍历列表中的每一个 JSON 对象 (每一条数据)
    for record in data:
        # 将单个 JSON 对象转换为字符串，并确保中文字符不被转义
        json_string = json.dumps(record, ensure_ascii=False)
        # 将这个字符串写入新文件，并在末尾添加一个换行符
        f_out.write(json_string + "\n")
        count += 1

print(f"处理完成！总共转换了 {count} 条数据。")
print(f"新的、格式正确的文件已保存为: {output_filename}")
print("请将这个新文件上传到 Google Colab。")
