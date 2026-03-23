import csv
from datasets import load_dataset
from tqdm import tqdm

# 换成你实际要读的数据集
ds = load_dataset("laion/laion2B-en-aesthetic", split="train", streaming=True)

output_file = "laion_url_caption_100.csv"
max_count = 100

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "global_caption"])  # 你要的表头

    count = 0
    pbar = tqdm(total=max_count, desc="导出进度", unit="条")

    for sample in ds:
        url = sample.get("url")
        caption = sample.get("caption") or sample.get("TEXT") or sample.get("text")

        if not url or not caption:
            continue

        writer.writerow([url, caption])
        count += 1
        pbar.update(1)

        if count >= max_count:
            break

    pbar.close()

print("导出完成：", output_file)