"""向量库管理:Embedding 模型与 FAISS 持久化。"""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config


def get_embeddings() -> HuggingFaceEmbeddings:
    """本地中文 embedding 模型,首次运行会自动下载(约 100MB)。"""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def build_vectorstore(chunks, vector_dir: Path = config.VECTOR_DIR) -> FAISS:
    """把文本块向量化并写入 FAISS,持久化到本地磁盘。"""
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    vector_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vector_dir))
    print(f"向量库已保存到 {vector_dir}")
    return vectorstore


def load_vectorstore(vector_dir: Path = config.VECTOR_DIR) -> FAISS:
    """加载已持久化的 FAISS 向量库;不存在时抛出带指引的异常。"""
    if not (vector_dir / "index.faiss").exists():
        raise FileNotFoundError(
            "向量库不存在,请先运行 `python ingest.py` 从 docs/ 下的课程资料构建"
        )
    # allow_dangerous_deserialization:索引文件由本项目 ingest.py 生成,来源可信
    return FAISS.load_local(str(vector_dir), get_embeddings(), allow_dangerous_deserialization=True)


def get_retriever(vector_dir: Path = config.VECTOR_DIR):
    """返回按相似度取 Top-K 的检索器。"""
    return load_vectorstore(vector_dir).as_retriever(search_kwargs={"k": config.RETRIEVER_K})
