# fix_format.py
import json

# --- 配置区 ---
INPUT_FILE = "finetune_data.jsonl"  # 你那个格式错误的文件名
OUTPUT_FILE = "finetune_data_FIXED.jsonl"  # 修复后的新文件名


def fix_finetune_data_format():
    print(f"--- 开始修复文件: {INPUT_FILE} ---")

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
            OUTPUT_FILE, "w", encoding="utf-8"
        ) as outfile:

            lines = infile.readlines()
            count = 0

            # 我们假设文件是成对出现的：一行user，一行assistant
            # 所以我们以步长为 2 来遍历
            for i in range(0, len(lines), 2):
                # 检查是否还有下一行，防止文件行数为奇数时出错
                if i + 1 < len(lines):
                    user_content = lines[i].strip()
                    assistant_content = lines[i + 1].strip()

                    # 确保两行都不是空的
                    if user_content and assistant_content:
                        # 构建正确的 messages 格式
                        messages = [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ]

                        # 包装成最终的 JSON 对象
                        final_json_obj = {"messages": messages}

                        # 写入新文件
                        outfile.write(
                            json.dumps(final_json_obj, ensure_ascii=False) + "\n"
                        )
                        count += 1

            if count > 0:
                print(f"\n✅ 修复完成！成功转换了 {count} 对数据。")
                print(f"新的、格式正确的文件已保存为: '{OUTPUT_FILE}'")
                print("现在，请将这个新文件上传到 Kaggle。")
            else:
                print(
                    "\n⚠️  警告：未能从输入文件中成功转换任何数据。请检查文件内容是否符合'一行user,一行assistant'的格式。"
                )

    except FileNotFoundError:
        print(
            f"❌ 错误：找不到输入文件 '{INPUT_FILE}'。请确保脚本和数据文件在同一个文件夹下。"
        )
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    fix_finetune_data_format()
