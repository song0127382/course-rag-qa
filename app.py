"""课程知识库问答助手 - Streamlit 界面
运行: streamlit run app.py
"""
import streamlit as st

import config
from rag.chain import build_rag_chain, retrieve_sources
from rag.loader import load_documents, split_documents
from rag.vectorstore import build_vectorstore

st.set_page_config(page_title="课程知识库问答助手", page_icon="📚", layout="centered")
st.title("📚 课程知识库问答助手")
st.caption("基于 LangChain + DeepSeek + FAISS 的 RAG 应用 · 回答严格依据你上传的课程资料,并附引用来源")

SUPPORTED = {".pdf", ".txt", ".md"}


def list_docs() -> list[str]:
    if not config.DOCS_DIR.exists():
        return []
    return sorted(p.name for p in config.DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED)


def has_vectorstore() -> bool:
    return (config.VECTOR_DIR / "index.faiss").exists()


@st.cache_resource(show_spinner="加载向量库中…")
def get_chain():
    return build_rag_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📚 知识库管理")

    files = list_docs()
    if files:
        st.markdown("**当前课件:**")
        for name in files:
            st.markdown(f"- {name}")
    else:
        st.info("还没有课件")

    uploaded = st.file_uploader(
        "上传课件(pdf / txt / md,可多选)", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("🔨 构建 / 重建知识库", type="primary", disabled=not (uploaded or files)):
        try:
            if uploaded:
                config.DOCS_DIR.mkdir(exist_ok=True)
                for up in uploaded:
                    (config.DOCS_DIR / up.name).write_bytes(up.getvalue())
            with st.spinner("加载 → 切分 → 向量化,正在构建…"):
                docs = load_documents()
                chunks = split_documents(docs)
                build_vectorstore(chunks)
            get_chain.clear()  # 重建后让问答链加载新向量库
            st.success(f"✅ 知识库已更新:{len(docs)} 份文档,共 {len(chunks)} 个文本块,可以开始提问了")
        except Exception as e:
            st.error(f"构建失败:{e}")

    st.divider()
    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.rerun()

if not has_vectorstore():
    st.warning("⚠️ 还没有知识库:请在左侧上传课件(pdf / txt / md),然后点击「🔨 构建 / 重建知识库」;也可以先用项目自带的《计算机网络》示例资料直接点击构建按钮体验。")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("📎 引用来源:" + "、".join(msg["sources"]))

question = st.chat_input("请输入你的问题,例如:什么是滑动窗口协议?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("检索课程资料并生成回答…"):
                chain = get_chain()
                answer = chain.invoke(question)
                sources = retrieve_sources(question)
            st.markdown(answer)
            st.caption("📎 引用来源:" + "、".join(sources))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"调用失败,请检查 .env 中的 DEEPSEEK_API_KEY 是否有效:{e}")
