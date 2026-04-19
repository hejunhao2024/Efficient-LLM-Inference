import os
from datasets import load_dataset

wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")

# 保存 test 集
with open("./data/WikiText-2/test.txt", "w", encoding="utf-8") as f:
    for item in wikitext["test"]:
        text = item["text"].strip()
        if text:  # 去掉空行
            f.write(text + "\n")

print("✅ WikiText-2 下载完成")

save_path = "/mnt/d/Labs/Efficient-LLM-Inference/data/PG-19"
os.makedirs(save_path, exist_ok=True)
dataset = load_dataset("emozilla/pg19", split="test", streaming=True)

sample = next(iter(dataset))
content = sample["text"]
book_title = sample.get("short_book_title", "sample_01").replace(" ", "_")

file_name = f"{book_title}.txt"
full_file_path = os.path.join(save_path, file_name)
with open(full_file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("-" * 30)
print(f"✅ 下载成功！")
