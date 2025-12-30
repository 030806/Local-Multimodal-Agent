import argparse
import os
import shutil
from modules.vector_store import VectorDBManager
from modules.classifier import SemanticClassifier
from modules.doc_processor import DocumentProcessor


def add_paper(args):
    """单篇论文处理逻辑 (封装为内部函数供批量处理调用)"""
    return _process_single_file(args.path, args.topics)


def _process_single_file(file_path, topics_str, db_manager=None, classifier=None, doc_processor=None):
    """内部核心逻辑：处理单份 PDF 文件"""
    topics = [t.strip() for t in topics_str.split(",")]

    # 如果没传入则初始化（批量处理时建议外部传入以复用模型加载）
    doc_processor = doc_processor or DocumentProcessor()
    classifier = classifier or SemanticClassifier()
    db_manager = db_manager or VectorDBManager()

    try:
        print(f"🔄 正在处理: {os.path.basename(file_path)} ...")
        # 读取并切片
        splits, first_page_text = doc_processor.load_and_split(file_path)

        # 语义分类
        category = classifier.classify_paper(first_page_text, topics)
        print(f"✅ 归类结果: [{category}]")

        # 移动文件
        new_path = doc_processor.move_file(file_path, category)

        # 更新元数据并存入向量库
        for split in splits:
            split.metadata['source'] = new_path
            split.metadata['category'] = category

        db_manager.add_documents(splits)
        return True
    except Exception as e:
        print(f"❌ 处理 {file_path} 出错: {e}")
        return False


def batch_process_papers(args):
    """批量处理文件夹中的所有 PDF"""
    if not os.path.exists(args.dir):
        print(f"❌ 错误：找不到目录 {args.dir}")
        return

    # 初始化管理器（在此初始化可实现模型复用，避免循环加载）
    doc_processor = DocumentProcessor()
    classifier = SemanticClassifier()
    db_manager = VectorDBManager()

    files = [f for f in os.listdir(args.dir) if f.lower().endswith('.pdf')]
    if not files:
        print(f"ℹ️ 在目录 {args.dir} 中未找到 PDF 文件。")
        return

    print(f"🚀 开始批量处理 {len(files)} 个文件...")
    success_count = 0
    for filename in files:
        full_path = os.path.join(args.dir, filename)
        if _process_single_file(full_path, args.topics, db_manager, classifier, doc_processor):
            success_count += 1

    print(f"\n✨ 批量整理完成！成功处理: {success_count}/{len(files)}")


def search_paper(args):
    """语义搜索文献 """
    query = args.query
    db_manager = VectorDBManager()
    k_val = 10 if args.index_only else 3
    print(f"🔍 正在搜索文献: '{query}' ...")
    results = db_manager.search_papers(query, k=k_val)

    if not results:
        print("❌ 未找到相关内容。")
        return

    print("\n" + "=" * 60)
    if args.index_only:
        seen_files = set()
        count = 0
        for doc in results:
            source_path = doc.metadata.get('source', 'Unknown')
            if source_path not in seen_files:
                count += 1
                print(f"{count}. 📄 {os.path.basename(source_path)}\n   路径: {source_path}")
                seen_files.add(source_path)
    else:
        for i, doc in enumerate(results):
            page_num = doc.metadata.get('page', 0) + 1
            print(f"🔎 结果 {i + 1} | 📄 {os.path.basename(doc.metadata.get('source', ''))}")
            print(f"📌 位置: 第 {page_num} 页 | 🏷️ 类别: {doc.metadata.get('category', 'Uncategorized')}")
            clean_content = doc.page_content.replace('\n', ' ')
            print(f"💬 片段: \"{clean_content[:250]}...\"")
            print("-" * 60)


def index_images(args):
    """图像索引 """
    db_manager = VectorDBManager()
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists(args.dir):
        print(f"❌ 错误：找不到目录 {args.dir}")
        return
    img_count = 0
    for root, _, files in os.walk(args.dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                if db_manager.add_image(full_path):
                    img_count += 1
    print(f"✨ 图像库更新完毕，共处理 {img_count} 张图片。")


def search_image(args):
    """图像搜索 (代码保持不变)"""
    db_manager = VectorDBManager()
    results = db_manager.search_images(args.query, k=3)
    if not results:
        print("❌ 未找到匹配图片。")
        return
    print(f"\n🔍 针对描述 '{args.query}' 的匹配结果:")
    print("=" * 60)
    for i, res in enumerate(results):
        similarity = max(0, 1 - (res['score'] / 2.0)) * 100
        print(f"结果 {i + 1} | 匹配度: {similarity:.2f}% (原始距离: {res['score']:.4f})")
        print(f"📁 路径: {res['path']}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Local AI Agent (Multi-modal)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. add_paper (单文件)
    add_p = subparsers.add_parser("add_paper")
    add_p.add_argument("path", type=str)
    add_p.add_argument("--topics", type=str, required=True)

    # 2. batch_process (批量文件夹)
    batch_p = subparsers.add_parser("batch_process")
    batch_p.add_argument("dir", type=str, help="Directory containing multiple PDFs")
    batch_p.add_argument("--topics", type=str, required=True)

    # 3. search_paper
    search_p = subparsers.add_parser("search_paper")
    search_p.add_argument("query", type=str)
    search_p.add_argument("--index-only", action="store_true")

    # 4. index_images
    idx_img_p = subparsers.add_parser("index_images")
    idx_img_p.add_argument("dir", type=str)

    # 5. search_image
    src_img_p = subparsers.add_parser("search_image")
    src_img_p.add_argument("query", type=str)

    args = parser.parse_args()

    if args.command == "add_paper":
        add_paper(args)
    elif args.command == "batch_process":
        batch_process_papers(args)
    elif args.command == "search_paper":
        search_paper(args)
    elif args.command == "index_images":
        index_images(args)
    elif args.command == "search_image":
        search_image(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()