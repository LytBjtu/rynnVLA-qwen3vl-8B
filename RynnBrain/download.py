# download_model.py
from huggingface_hub import snapshot_download

local_dir = "D:/RynnBrain/models/RynnBrain-2B-local"

snapshot_download(
    repo_id="Alibaba-DAMO-Academy/RynnBrain-2B",
    local_dir=local_dir,
    endpoint="https://hf-mirror.com",  # 使用镜像
    ignore_patterns=["*.git*", "*.md", "LICENSE"],  # 可选
)