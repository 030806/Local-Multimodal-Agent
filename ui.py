import streamlit as st
import os
from PIL import Image
from modules.vector_store import VectorDBManager
from modules.classifier import SemanticClassifier
from modules.doc_processor import DocumentProcessor

# 设置页面配置
st.set_page_config(page_title="Local Multimodal AI Agent", page_icon="🤖", layout="wide")


# 初始化后端组件 (使用 st.cache_resource 避免重复加载模型)
@st.cache_resource
def get_managers():
    return VectorDBManager(), SemanticClassifier(), DocumentProcessor()


db_manager, classifier, doc_processor = get_managers()

# --- 侧边栏导航 ---
st.sidebar.title("🤖 导航控制台")
menu = st.sidebar.radio("选择功能模块", [
    "🏠 首页",
    "📄 文献上传与整理",
    "📂 批量论文整理",
    "🔍 文献语义搜索",
    "🖼️ 图像库搜索"
])

st.sidebar.markdown("---")
st.sidebar.info("项目状态：已连接本地 CLIP & MiniLM 模型")

# --- 1. 首页 ---
if menu == "🏠 首页":
    st.title("欢迎使用本地 AI 智能管理助手")
    st.markdown("""
    本项目利用多模态神经网络技术，为您提供：
    - **智能文献管理**：自动分析 PDF 主题并归档，支持全文语义搜索。
    - **智能图像管理**：利用 CLIP 模型，实现“以文搜图”。
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("文档索引数", "已就绪")
    with col2:
        st.metric("图像索引数", "已就绪")

# --- 2. 文献上传与整理 ---
elif menu == "📄 文献上传与整理":
    st.header("📄 上传新论文")

    uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")
    topics_input = st.text_input("定义分类主题 (逗号分隔)", "NLP, Computer Vision, Reinforcement Learning,Deep Learning")

    if st.button("开始处理并归类"):
        if uploaded_file and topics_input:
            with st.spinner("🚀 正在提取文本并进行语义分类..."):
                # 保存临时文件以便处理
                temp_path = os.path.join("test_data/papers", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                topics = [t.strip() for t in topics_input.split(",")]

                # 执行后端逻辑
                splits, first_page_text = doc_processor.load_and_split(temp_path)
                category = classifier.classify_paper(first_page_text, topics)

                # 移动并更新数据库
                new_path = doc_processor.move_file(temp_path, category)
                for split in splits:
                    split.metadata['source'] = new_path
                    split.metadata['category'] = category
                db_manager.add_documents(splits)

                st.success(f"✅ 文件已自动归类至: **[{category}]**")
                st.balloons()
        else:
            st.warning("请上传文件并输入主题。")

# --- 3. 文献语义搜索 ---
elif menu == "🔍 文献语义搜索":
    st.header("🔍 文献深度搜索")
    query = st.text_input("输入您的疑问 (例如: How does attention mechanism work?)")
    index_only = st.checkbox("仅返回文件索引")

    if st.button("搜索"):
        if query:
            k = 10 if index_only else 3
            results = db_manager.search_papers(query, k=k)

            if results:
                if index_only:
                    seen = set()
                    for doc in results:
                        path = doc.metadata.get('source', 'Unknown')
                        if path not in seen:
                            st.write(f"📄 **{os.path.basename(path)}**")
                            st.caption(f"路径: {path}")
                            seen.add(path)
                else:
                    for i, doc in enumerate(results):
                        with st.expander(
                                f"结果 {i + 1}: {os.path.basename(doc.metadata.get('source', ''))} (第 {doc.metadata.get('page', 0) + 1} 页)"):
                            st.write(f"**分类标签:** :blue[{doc.metadata.get('category', 'N/A')}]")
                            st.write(f"**片段内容:** ...{doc.page_content}...")
            else:
                st.error("未找到匹配内容。")

# --- 4. 图像库搜索 ---
elif menu == "🖼️ 图像库搜索":
    st.header("🖼️ 智能图像管理")

    # --- 第一部分：索引构建 (Indexing) ---
    with st.expander("🛠️ 图像索引维护", expanded=False):
        st.write("如果这是您第一次使用或更换了图片目录，请先进行索引。")
        img_dir = st.text_input("图像文件夹路径", value="./test_data/images")

        if st.button("开始构建/更新图像索引"):
            if os.path.exists(img_dir):
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                files = [f for f in os.listdir(img_dir) if f.lower().endswith(image_extensions)]

                if not files:
                    st.warning("该目录下没有发现图片文件。")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    count = 0

                    for i, filename in enumerate(files):
                        full_path = os.path.join(img_dir, filename)
                        status_text.text(f"正在索引: {filename}")
                        if db_manager.add_image(full_path):
                            count += 1
                        progress_bar.progress((i + 1) / len(files))

                    st.success(f"✨ 索引完成！已成功索引 {count} 张图片。")
            else:
                st.error("路径不存在，请检查。")

    st.markdown("---")

    # --- 第二部分：以文搜图 (Search) ---
    st.subheader("🔍 以文搜图 (CLIP Search)")
    img_query = st.text_input("输入描述词 (例如: a photo of a dog, sunset, paper chart)")
    top_k = st.slider("返回结果数量", 1, 10, 3)

    if st.button("搜索图片"):
        if img_query:
            with st.spinner("🧠 CLIP 正在理解语义..."):
                results = db_manager.search_images(img_query, k=top_k)

            if results:
                st.write(f"为您找到以下 {len(results)} 张最匹配的图片：")
                cols = st.columns(3)
                for idx, res in enumerate(results):
                    with cols[idx % 3]:
                        similarity = max(0, 1 - (res['score'] / 2.0)) * 100
                        # 将 use_column_width=True 替换为 use_container_width=True
                        st.image(res['path'], use_container_width=True)
                        st.caption(f"🎯 匹配度: {similarity:.2f}%")
                        st.caption(f"📂 `{os.path.basename(res['path'])}`")
            else:
                st.info("💡 未找到匹配图片。请确保已先执行上方‘索引维护’功能。")
elif menu == "📂 批量论文整理":
    st.header("📂 一键整理论文文件夹")
    st.info("系统将扫描指定文件夹下的所有 PDF，自动进行语义分类、移动文件并建立索引。")

    source_dir = st.text_input("请输入待整理的文件夹路径 (例如: ./test_data/raw_papers)")
    batch_topics = st.text_input("分类主题 (逗号分隔)", "NLP, Computer Vision, Reinforcement Learning,Deep Learning")

    if st.button("开始批量整理"):
        if not os.path.exists(source_dir):
            st.error("❌ 路径不存在，请检查后重试。")
        else:
            # 获取所有待处理的 PDF
            pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]

            if not pdf_files:
                st.warning("查无 PDF 文件。")
            else:
                st.write(f"🔍 发现 {len(pdf_files)} 个待处理文件...")

                # 初始化进度条和日志占位符
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_area = st.expander("详细处理日志", expanded=True)

                topics = [t.strip() for t in batch_topics.split(",")]
                success_count = 0

                for i, filename in enumerate(pdf_files):
                    file_path = os.path.join(source_dir, filename)
                    status_text.text(f"正在处理 ({i + 1}/{len(pdf_files)}): {filename}")

                    try:
                        # 1. 加载与切片
                        splits, first_page_text = doc_processor.load_and_split(file_path)

                        # 2. 语义分类
                        category = classifier.classify_paper(first_page_text, topics)

                        # 3. 移动文件
                        new_path = doc_processor.move_file(file_path, category)

                        # 4. 存入数据库
                        for split in splits:
                            split.metadata['source'] = new_path
                            split.metadata['category'] = category
                        db_manager.add_documents(splits)

                        log_area.write(f"✅ {filename} -> **[{category}]**")
                        success_count += 1

                    except Exception as e:
                        log_area.error(f"❌ {filename} 处理失败: {str(e)}")

                    # 更新进度条
                    progress_bar.progress((i + 1) / len(pdf_files))

                st.success(f"✨ 批量整理完成！成功处理 {success_count} 个文件。")
                st.balloons()