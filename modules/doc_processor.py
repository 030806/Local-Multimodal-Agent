import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.config import DOCS_DIR


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    def load_and_split(self, file_path):
        """读取 PDF 并切分为用于搜索的片段"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splits = self.text_splitter.split_documents(docs)
        return splits, docs[0].page_content  # 返回切片用于存储，返回第一页内容用于分类

    def move_file(self, file_path, category):
        """将文件移动到对应的分类文件夹"""
        target_dir = os.path.join(DOCS_DIR, category)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        filename = os.path.basename(file_path)
        target_path = os.path.join(target_dir, filename)

        shutil.move(file_path, target_path)
        print(f"📂 文件已归档至: {target_path}")
        return target_path