"""构建向量库入口:python ingest.py
读取 docs/ 下的课程资料 -> 切分 -> 向量化 -> 保存到 data/vectorstore/
"""
from rag.loader import load_documents, split_documents
from rag.vectorstore import build_vectorstore


def main():
    print("=== 1/3 加载课程资料 ===")
    docs = load_documents()
    print("=== 2/3 切分文本块 ===")
    chunks = split_documents(docs)
    print("=== 3/3 向量化并保存 ===")
    build_vectorstore(chunks)
    print("完成!现在可以运行 `streamlit run app.py` 开始提问")


if __name__ == "__main__":
    main()
