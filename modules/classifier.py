from sentence_transformers import SentenceTransformer, util
from modules.config import EMBEDDING_MODEL_PATH
import torch
import re
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt_tab')
# except LookupError:
#     print("📥 正在下载缺失的 nltk 资源: punkt_tab...")
#     nltk.download('punkt_tab')
# ----------------------

class SemanticClassifier:
    def __init__(self):
        print(f"🔄 正在加载分类模型: {EMBEDDING_MODEL_PATH} ...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    def _clean_text(self, text):
        """
        激进清洗：只保留最能体现学科特征的词汇。
        """
        # 1. 统一转小写
        text = text.lower()

        # 2. 尝试定位 Abstract 关键词，因为摘要最有代表性
        abstract_pos = text.find("abstract")
        if abstract_pos != -1:
            # 取摘要开始后的 1200 个字符
            text = text[abstract_pos:abstract_pos + 1200]
        else:
            # 如果没找到 Abstract，取前 1500 个字符（避开最顶部的作者学校信息）
            text = text[200:1500]

        # 3. 移除干扰项：移除常见的学校名称、邮箱后缀、日期等噪音
        text = re.sub(r'\S+@\S+', '', text)  # 移除邮箱
        text = re.sub(r'http\S+', '', text)  # 移除链接

        return text

    def classify_paper(self, text_content, topics):
        # 1. 文本清洗：只取摘要部分，减少噪音
        input_text = self._clean_text(text_content)

        # 2. 语义增强策略：为每个主题定义“特征词群”
        topic_enhancement = {
            "NLP": "natural language processing, Natural Language Processing, NLP, text sequences, translation, "
                   "vocabulary, linguistics, transformer, bert, word embedding,language model,llm,text generation,"
                   "machine translation,question answering,dialogue,information extraction,sentiment",
            "Computer Vision": "Computer Vision, CV, image recognition, object detection, pixel, convolutional neural networks, CNN, ResNet, vision, video,3d vision,anomaly detection,image segmentation,image classification",
            "Reinforcement Learning": "Reinforcement Learning, RL, agent, reward, policy gradient, MDP, environment, Q-learning, action space,game theory",
            "Deep Learning":"neural network,cnn,rnn,lstm,transformer,attention,gan,diffusion,autoencoder,gnn,graph neural",

        }

        # 构建对比向量：优先使用增强词群，如果没有则用原词
        enhanced_topics = [topic_enhancement.get(t, t) for t in topics]

        # 3. 计算向量
        text_embedding = self.model.encode(input_text, convert_to_tensor=True)
        topic_embeddings = self.model.encode(enhanced_topics, convert_to_tensor=True)

        # 4. 计算余弦相似度
        cosine_scores = util.cos_sim(text_embedding, topic_embeddings)[0]

        # 打印调试信息，看每个主题的得分
        for i, t in enumerate(topics):
            print(f"DEBUG: 主题 [{t}] 得分: {cosine_scores[i].item():.4f}")

        best_score_idx = torch.argmax(cosine_scores).item()
        return topics[best_score_idx]