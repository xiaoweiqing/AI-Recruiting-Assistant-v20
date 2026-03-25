# validate_data.py
import json


def validate_jsonl_file(file_path):
    print(f"--- 开始校验文件: {file_path} ---")
    problematic_lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            # 检查是否为空行
            if not line.strip():
                print(f"⚠️  警告 - 第 {i} 行是空行。")
                problematic_lines.append(i)
                continue

            try:
                # 尝试解析 JSON
                data = json.loads(line)

                # 检查核心结构
                if "messages" not in data:
                    print(f"❌ 错误 - 第 {i} 行: 缺少 'messages' 键。")
                    problematic_lines.append(i)
                    continue
                if not isinstance(data["messages"], list):
                    print(f"❌ 错误 - 第 {i} 行: 'messages' 的值不是一个列表。")
                    problematic_lines.append(i)
                    continue
                if len(data["messages"]) == 0:
                    print(f"❌ 错误 - 第 {i} 行: 'messages' 列表是空的。")
                    problematic_lines.append(i)
                    continue

            except json.JSONDecodeError:
                print(f"❌ 错误 - 第 {i} 行: 不是有效的 JSON 格式。")
                problematic_lines.append(i)

    if not problematic_lines:
        print("\n✅ 恭喜！文件格式完全正确，没有发现任何问题！")
    else:
        print(f"\n❗️❗️❗️ 文件校验完成，在以下行发现了 {len(problematic_lines)} 个问题:")
        print(problematic_lines)
        print("请打开文件，找到这些行并修正它们的格式。")


if __name__ == "__main__":
    validate_jsonl_file("finetune_data.jsonl")
