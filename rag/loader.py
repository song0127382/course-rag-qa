"""文档加载与切分:把 docs/ 下的课程资料读入并切成文本块。"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import config

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_documents(docs_dir: Path = config.DOCS_DIR) -> list[Document]:
    """加载目录下所有支持的文档,并在元数据中记录来源文件名。"""
    all_docs: list[Document] = []
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(
            f"目录 {docs_dir} 下没有可用的课程资料,请放入 .pdf / .txt / .md 文件"
        )
    for path in files:
        loader = PyPDFLoader(path) if path.suffix.lower() == ".pdf" else TextLoader(path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path.name
        all_docs.extend(docs)
        print(f"已加载 {path.name}:{len(docs)} 页/段")
    return all_docs


def split_documents(docs: list[Document]) -> list[Document]:
    """把长文档切成带重叠的小块,便于向量检索。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"共切分为 {len(chunks)} 个文本块")
    return chunks
