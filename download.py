import os
from huggingface_hub import snapshot_download

# 走 hf-mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模型仓库
repo_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

# 本地保存路径
local_dir = r"D:\Project\models\llava-interleave-qwen-0.5b-hf"

# 下载
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Windows 建议关闭软链接
    resume_download=True  # 断点续传
)

print("下载完成，保存路径为：", local_dir)