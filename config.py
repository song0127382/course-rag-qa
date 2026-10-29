"""全局配置:模型、路径与检索参数。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 国内网络默认走 HuggingFace 镜像下载 embedding 模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录与数据目录
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"          # 课程资料目录(支持 pdf / txt / md)
VECTOR_DIR = BASE_DIR / "data" / "vectorstore"  # FAISS 向量库持久化目录

# Embedding 模型:本地开源中文模型 bge-small-zh,无需调用任何在线接口
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 生成模型:DeepSeek 在线 API
CHAT_MODEL = "deepseek-chat"

# 文本切分与检索参数
CHUNK_SIZE = 500        # 每个文本块的目标长度(字符)
CHUNK_OVERLAP = 50      # 相邻块的重叠长度,避免语义被切断
RETRIEVER_K = 4         # 每次检索返回的最相关文本块数量
