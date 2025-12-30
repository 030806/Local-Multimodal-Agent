import os

# 路径设置
BASE_DIR = './agent'
DOCS_DIR = os.path.join('./', "documents")
DB_DIR = os.path.join('./', "db")

# ===================================================
# 模型配置
# ===================================================

# 1. Embedding 模型路径 (填入刚才 download_models.py 输出的绝对路径)
# 注意：Windows下路径分隔符是 \\ 或 /
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "AI-ModelScope", "all-MiniLM-L6-v2")

# 在 config.py 中添加 CLIP 模型用于图像和文本的跨模态匹配
CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_MODEL_PATH = "./agent/models/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
IMG_DIR = os.path.join('./', "images") # 存放图片的目录