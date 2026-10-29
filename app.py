"""课程知识库问答助手 - 「小课」私人助教界面
运行: streamlit run app.py
"""
import streamlit as st

import config
from rag.chain import build_rag_chain, retrieve_sources
from rag.loader import load_documents, split_documents
from rag.vectorstore import build_vectorstore

st.set_page_config(page_title="小课 · 你的私人课程助教", page_icon="🦉", layout="centered")

# ── 吉祥物「小课」:戴学士帽的猫头鹰(SVG 手绘,无外部依赖)──────────────
OWL_SVG = """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="24" cy="74" rx="9" ry="21" fill="#4A6AA5"/>
  <ellipse cx="96" cy="74" rx="9" ry="21" fill="#4A6AA5"/>
  <ellipse cx="60" cy="72" rx="38" ry="42" fill="#5B7DB1"/>
  <ellipse cx="60" cy="84" rx="24" ry="26" fill="#EAF1FB"/>
  <circle cx="46" cy="60" r="14" fill="#FFFFFF"/>
  <circle cx="74" cy="60" r="14" fill="#FFFFFF"/>
  <circle cx="47" cy="61" r="6" fill="#2B3A55"/>
  <circle cx="73" cy="61" r="6" fill="#2B3A55"/>
  <circle cx="49" cy="59" r="2" fill="#FFFFFF"/>
  <circle cx="75" cy="59" r="2" fill="#FFFFFF"/>
  <polygon points="60,68 53,77 67,77" fill="#F4A259"/>
  <polygon points="60,14 94,29 60,37 26,29" fill="#39476B"/>
  <rect x="36" y="27" width="48" height="7" rx="3" fill="#2B3A55"/>
  <line x1="90" y1="31" x2="90" y2="44" stroke="#F4A259" stroke-width="2.5"/>
  <circle cx="90" cy="47" r="3.2" fill="#F4A259"/>
</svg>"""

CUSTOM_CSS = """<style>
.block-container { padding-top: 1.4rem; max-width: 860px; }
.hero { display: flex; align-items: center; gap: 22px;
        background: linear-gradient(135deg, #4F6DF5 0%, #7B5CF0 100%);
        border-radius: 22px; padding: 24px 30px; color: #fff;
        margin-bottom: 20px; box-shadow: 0 8px 22px rgba(79,109,245,.28); }
.hero-avatar { width: 88px; height: 88px; background: #fff; border-radius: 50%;
               padding: 9px; flex-shrink: 0; box-shadow: 0 3px 10px rgba(0,0,0,.18); }
.hero-avatar svg { width: 100%; height: 100%; display: block; }
.hero-title { font-size: 1.4rem; font-weight: 700; letter-spacing: .5px; }
.hero-sub { font-size: .95rem; opacity: .93; margin-top: 7px; }
[data-testid="stChatMessage"] { border-radius: 16px; padding: 12px 16px;
        background: #FFFFFF; border: 1px solid #EEF0F6;
        box-shadow: 0 1px 6px rgba(30,40,80,.07); }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #EDF1FF; border-color: #DCE3FF; }
[data-testid="stSidebar"] { background: #F7F8FC; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: .92rem; }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #4F6DF5, #7B5CF0); border: none; color: #fff; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { filter: brightness(1.1); }
.side-mascot { display: flex; align-items: center; gap: 12px;
        background: #fff; border-radius: 16px; padding: 12px;
        border: 1px solid #EEF0F6; margin-bottom: 14px; }
.side-mascot svg { width: 52px; height: 52px; flex-shrink: 0; }
.side-mascot .t { font-weight: 700; font-size: .98rem; color: #2B3A55; }
.side-mascot .s { font-size: .8rem; color: #7A8599; }
footer { visibility: hidden; }
</style>"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
<div class="hero">
  <div class="hero-avatar">{OWL_SVG}</div>
  <div>
    <div class="hero-title">嗨,我是小课,你的私人课程助教 🦉</div>
    <div class="hero-sub">课件我已通读完毕 · 随时开问 · 每句回答都出自你的资料,并附引用出处</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

SUPPORTED = {".pdf", ".txt", ".md"}

SUGGESTIONS = [
    ("什么是三次握手?", "什么是三次握手?为什么需要三次而不是两次?"),
    ("TCP vs UDP 区别", "TCP 和 UDP 有什么区别?"),
    ("交换机自学习原理", "以太网交换机是如何自学习的?"),
]


def list_docs() -> list[str]:
    if not config.DOCS_DIR.exists():
        return []
    return sorted(p.name for p in config.DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED)


def has_vectorstore() -> bool:
    return (config.VECTOR_DIR / "index.faiss").exists()


@st.cache_resource(show_spinner="小课正在翻开课件…")
def get_chain():
    return build_rag_chain()


def ask(question: str):
    """处理一次提问:展示双方气泡并写入历史。"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🦉"):
        try:
            with st.spinner("小课正在翻课件找答案…"):
                answer = get_chain().invoke(question)
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


# ── 侧边栏:小课的知识库管理 ────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""<div class="side-mascot">{OWL_SVG}
        <div><div class="t">小课的知识库</div><div class="s">上传课件喂养我</div></div></div>""",
        unsafe_allow_html=True,
    )

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
            st.success(f"✅ 已更新:{len(docs)} 份文档,{len(chunks)} 个文本块,来问我吧!")
        except Exception as e:
            st.error(f"构建失败:{e}")

    st.divider()
    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.rerun()

# ── 主对话区 ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not has_vectorstore():
    st.warning(
        "⚠️ 还没有知识库:请在左侧上传课件(pdf / txt / md),然后点击「🔨 构建 / 重建知识库」;"
        "也可以直接用项目自带的《计算机网络》示例课件,点构建按钮即可体验。"
    )

if not st.session_state.messages:
    # 小课主动开口,以对话开启教学
    with st.chat_message("assistant", avatar="🦉"):
        st.markdown(
            "嗨!我是**小课** 🦉 你这门课的专属助教。\n\n"
            "课件我已经通读完了,你只管问——我保证:每句话都出自课件原文,资料里没有的我会直说,绝不瞎编。\n\n"
            "**不知道从哪开始?点一个话题,我马上开讲 👇**"
        )
        cols = st.columns(len(SUGGESTIONS))
        for col, (label, full_question) in zip(cols, SUGGESTIONS):
            if col.button(label, use_container_width=True):
                ask(full_question)
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🦉"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("📎 引用来源:" + "、".join(msg["sources"]))

question = st.chat_input("向小课提问,例如:什么是滑动窗口协议?")
if question:
    ask(question)
