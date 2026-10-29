"""课程知识库问答助手 - Streamlit 界面
运行: streamlit run app.py
"""
import streamlit as st

from rag.chain import build_rag_chain, retrieve_sources

st.set_page_config(page_title="课程知识库问答助手", page_icon="📚", layout="centered")
st.title("📚 课程知识库问答助手")
st.caption("基于 LangChain + DeepSeek + FAISS 的 RAG 应用 · 回答依据本地课程资料生成")

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource(show_spinner="加载向量库中…")
def get_chain():
    return build_rag_chain()


with st.sidebar:
    st.header("使用说明")
    st.markdown(
        "1. 把课程资料(pdf / txt / md)放入 `docs/` 目录\n"
        "2. 运行 `python ingest.py` 构建向量库\n"
        "3. 在下方输入框提问,回答会附引用来源"
    )
    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.rerun()

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
