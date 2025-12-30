import re
from langchain_community.llms import Ollama
from modules.config import LLM_MODEL_NAME


class LLMClassifier:
    def __init__(self):
        self.llm = Ollama(model=LLM_MODEL_NAME)

    def classify_paper(self, text_content, topics):
        prompt = f"""
        你是一个专业的学术助手。请阅读以下论文摘要，并将其归类到以下主题列表中：{topics}。

        论文摘要：
        {text_content[:5000]}

        要求：
        1. 仅输出一个最匹配的主题单词。
        2. 不要输出思考过程，不要解释。
        """

        try:
            response = self.llm.invoke(prompt)

            # --- 针对 DeepSeek R1 的清洗逻辑 ---
            # 去除 <think>...</think> 标签及其内容
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            # ------------------------------------

            # 清洗多余空格和符号
            category = response.strip().replace('"', '').replace("'", "").replace("**", "")

            # 匹配有效主题
            for topic in topics:
                if topic.lower() in category.lower():
                    return topic

            # 如果没匹配上，打印出来看看模型到底回了什么（方便调试）
            print(f"⚠️ 模型原始回复: {category}")
            return "Uncategorized"

        except Exception as e:
            print(f"❌ LLM 分类失败: {e}")
            return "Uncategorized"