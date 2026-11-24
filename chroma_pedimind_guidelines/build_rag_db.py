import os
import gc
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================= 配置区 =================
# 你的 PDF 文件列表（请确保文件名和路径完全一致）
PDF_FILES = [
    "诸福棠实用儿科学（第九版上-1）.pdf",
    "诸福棠实用儿科学（第九版上-2）.pdf",
    "诸福棠实用儿科学（第九版上-3）.pdf",
    "诸福棠实用儿科学（第九版上-4）.pdf",
    "诸福棠实用儿科学（第九版上-5）.pdf",
]

# 你原有的 txt 文件（如果不需要，可以注释掉）
TXT_FILES = ["medical_guidelines.txt"]

# 向量库持久化目录
PERSIST_DIR = "./chroma_pedimind_guidelines"
COLLECTION_NAME = "pedimind_guidelines"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 800  # 每次写入 Chroma 的块数
# ==========================================

def get_embeddings():
    print(f"正在加载 Embedding 模型: {EMBEDDING_MODEL}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vectorstore():
    embeddings = get_embeddings()
    
    # 1. 初始化分块器
    # 医学书籍建议 chunk_size 设为 600 左右，保证完整的病理/临床表现描述不被切断
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    # 2. 初始化空的持久化向量库（不存在则创建，存在则直接追加）
    print(f"正在初始化向量库 (存储路径: {PERSIST_DIR})...")
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    total_chunks = 0

    # 3. 处理 PDF 文件
    for pdf_path in PDF_FILES:
        if not os.path.exists(pdf_path):
            print(f"⚠️ 找不到文件，跳过: {pdf_path}")
            continue

        print(f"\n📄 开始加载: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        
        chunks = splitter.split_documents(pages)
        print(f"  → 切分出 {len(chunks)} 个文本块")
        
        # 分批写入防止内存撑爆或 SQLite 限制
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vectordb.add_documents(batch)
            total_chunks += len(batch)
            print(f"  → 已写入 {total_chunks} 块 (当前 PDF)")
        
        # 释放内存
        del pages, chunks
        gc.collect()

    # 4. 处理 TXT 文件（你原有的指南）
    for txt_path in TXT_FILES:
        if not os.path.exists(txt_path):
            print(f"⚠️ 找不到文件，跳过: {txt_path}")
            continue
            
        print(f"\n📄 开始加载: {txt_path}")
        loader = TextLoader(txt_path, encoding="utf-8")
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vectordb.add_documents(batch)
            total_chunks += len(batch)
        print(f"  → TXT 处理完成，新增 {len(chunks)} 块")

    # 5. 统一持久化
    vectordb.persist()
    print(f"\n✅ 构建完成！共处理 {total_chunks} 个文本块。")
    print(f"📁 数据已保存至：{PERSIST_DIR}")

if __name__ == "__main__":
    build_vectorstore()