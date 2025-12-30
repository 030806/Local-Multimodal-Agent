# download_models.py
import os
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download

# 设置模型保存的根目录 (你可以改成你自己的路径，例如 '/data/ljf/LLM/models/...')
CACHE_DIR = './agent/models'
print(f"🚀 准备下载模型到: {CACHE_DIR}")

# 1. 下载 Embedding 模型 (用于文档搜索)
# 对应 HuggingFace 的 sentence-transformers/all-MiniLM-L6-v2

print("正在从 ModelScope 下载 CLIP 模型...")
# 下载 OpenAI 开源的 CLIP ViT-B-32
# clip_path = snapshot_download(
#     'openai/clip-vit-base-patch32',
#     cache_dir=CACHE_DIR
# )
clip_path = hf_snapshot_download(
    repo_id='openai/clip-vit-base-patch32',
    cache_dir=CACHE_DIR,
    resume_download=True  # 支持断点续传
)
print(f"✅ CLIP 模型已下载至: {clip_path}")

print("正在下载 Embedding 模型 (all-MiniLM-L6-v2)...")
embedding_path = snapshot_download(
    'AI-ModelScope/all-MiniLM-L6-v2',
    cache_dir=CACHE_DIR
)
print(f"✅ Embedding 模型已下载: {embedding_path}")

# 2. (可选) 如果你也想下载 DeepSeek 的原始权重 (非 Ollama 版)
# 如果你已经用 Ollama 跑起来了，这一步可以跳过
# print("正在下载 DeepSeek 模型...")
# llm_path = snapshot_download(
#     'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
#     cache_dir=CACHE_DIR
# )
# print(f"✅ LLM 模型已下载: {llm_path}")

