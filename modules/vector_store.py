import os
import torch
import chromadb
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from modules.config import DB_DIR, EMBEDDING_MODEL_PATH, CLIP_MODEL_PATH


class VectorDBManager:
    def __init__(self):
        # 1. 检查并设置设备 (GPU/CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 使用设备: {self.device}")

        # 2. 初始化文献 Embedding 模型 (纯文本)
        print(f"🔄 正在加载文档 Embedding 模型: {os.path.basename(EMBEDDING_MODEL_PATH)}...")
        self.doc_embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

        # 3. 初始化原生 CLIP 模型 (多模态)
        print(f"🔄 正在通过 Transformers 加载 CLIP 模型...")

        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(self.device)
        # self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH, use_fast=True)

        # 4. 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=DB_DIR)

        # 文档 Collection
        self.paper_db = Chroma(
            client=self.client,
            collection_name="paper_collection",
            embedding_function=self.doc_embedder
        )

        # 图像 Collection
        self.image_col = self.client.get_or_create_collection(name="image_collection")

    # ================= 智能图像管理模块 (2.2) =================

    def add_image(self, img_path):
        """生成图像 Embedding 并存入库"""
        try:
            image = Image.open(img_path).convert("RGB")

            # 使用 CLIPProcessor 预处理图像并生成 Embedding
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.clip_model.get_image_features(**inputs)
                # 归一化特征向量
                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_embedding = image_features.cpu().numpy().flatten().tolist()

            self.image_col.add(
                embeddings=[img_embedding],
                documents=[img_path],
                metadatas=[{"file_path": img_path}],
                ids=[os.path.basename(img_path)]
            )
            return True
        except Exception as e:
            print(f"❌ 图片处理失败 {img_path}: {e}")
            return False

    def search_images(self, query_text, k=3):
        """以文搜图：带有 Prompt Template 优化的检索"""
        try:
            # 1. 优化提示词：如果用户没输入 a photo of，我们自动补上
            # 这样可以更好地激活 CLIP 在预训练时学到的视觉特征
            if not query_text.lower().startswith("a photo of"):
                optimized_query = f"a photo of a {query_text}"
            else:
                optimized_query = query_text

            print(f"🪄 优化后的 Query: '{optimized_query}'")

            with torch.no_grad():
                # 使用 CLIPProcessor 处理优化后的搜索文本
                inputs = self.clip_processor(
                    text=[optimized_query],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                text_features = self.clip_model.get_text_features(**inputs)
                # 归一化
                text_features /= text_features.norm(dim=-1, keepdim=True)
                query_embedding = text_features.cpu().numpy().flatten().tolist()

            results = self.image_col.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "path": results['documents'][0][i],
                        "score": results['distances'][0][i]
                    })
            return formatted_results
        except Exception as e:
            print(f"❌ 图像检索失败: {e}")
            return []
    # def search_images(self, query_text, k=3):
    #     """以文搜图：通过 CLIP 文本分支检索图像"""
    #     try:
    #         with torch.no_grad():
    #             # 使用 CLIPProcessor 处理搜索文本
    #             inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
    #             text_features = self.clip_model.get_text_features(**inputs)
    #             # 归一化
    #             text_features /= text_features.norm(dim=-1, keepdim=True)
    #             query_embedding = text_features.cpu().numpy().flatten().tolist()
    #
    #         results = self.image_col.query(
    #             query_embeddings=[query_embedding],
    #             n_results=k
    #         )
    #
    #         formatted_results = []
    #         if results['documents']:
    #             for i in range(len(results['documents'][0])):
    #                 formatted_results.append({
    #                     "path": results['documents'][0][i],
    #                     "score": results['distances'][0][i]
    #                 })
    #
    #         return formatted_results
    #     except Exception as e:
    #         print(f"❌ 图像检索失败: {e}")
    #         return []

    # ================= 文献管理模块 (2.1) =================

    def add_documents(self, documents):
        """将 PDF 切片存入文档库"""
        self.paper_db.add_documents(documents)
        print(f"✅ 已将 {len(documents)} 个文献片段存入数据库。")

    def search_papers(self, query, k=3):
        """语义搜索文献"""
        return self.paper_db.similarity_search(query, k=k)