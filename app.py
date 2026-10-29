"""课程知识库问答助手 - 「苏老师」私人助教界面(黑板粉笔风)
运行: streamlit run app.py
"""
import streamlit as st

import config
from rag.chain import build_rag_chain, retrieve_sources
from rag.loader import load_documents, split_documents
from rag.vectorstore import build_vectorstore

st.set_page_config(page_title="苏老师 · 你的私人课程助教", page_icon="🎓", layout="centered")

# ── 人物形象「苏老师」:盘发、圆框眼镜、深绿西装、手持教材(SVG 手绘,无外部依赖)──
TEACHER_SVG = """<svg viewBox="0 0 200 210" xmlns="http://www.w3.org/2000/svg">
  <path d="M58 210 L63 148 Q68 118 100 118 Q132 118 137 148 L142 210 Z" fill="#2F5D50"/>
  <path d="M86 120 L100 152 L114 120 Q107 116 100 116 Q93 116 86 120 Z" fill="#F5F1E6"/>
  <path d="M86 120 L100 152 L92 154 L80 128 Z" fill="#274C42"/>
  <path d="M114 120 L100 152 L108 154 L120 128 Z" fill="#274C42"/>
  <circle cx="100" cy="160" r="4" fill="#E8B04B"/>
  <g transform="rotate(-8 70 182)">
    <rect x="44" y="168" width="48" height="30" rx="4" fill="#D9A03F"/>
    <rect x="48" y="172" width="40" height="22" rx="2" fill="#FDF8EC"/>
    <line x1="68" y1="172" x2="68" y2="194" stroke="#D9C48E" stroke-width="2"/>
  </g>
  <circle cx="100" cy="60" r="40" fill="#5A4634"/>
  <circle cx="70" cy="84" r="12" fill="#5A4634"/>
  <circle cx="130" cy="84" r="12" fill="#5A4634"/>
  <circle cx="100" cy="16" r="15" fill="#5A4634"/>
  <circle cx="100" cy="16" r="8" fill="#6E5743"/>
  <circle cx="100" cy="72" r="31" fill="#FFD9B3"/>
  <path d="M69 66 Q74 40 100 40 Q126 40 131 66 Q116 52 100 54 Q84 52 69 66 Z" fill="#5A4634"/>
  <circle cx="87" cy="74" r="10.5" fill="#FFFFFF" fill-opacity=".35" stroke="#2B3A55" stroke-width="2.5"/>
  <circle cx="113" cy="74" r="10.5" fill="#FFFFFF" fill-opacity=".35" stroke="#2B3A55" stroke-width="2.5"/>
  <line x1="97.5" y1="74" x2="102.5" y2="74" stroke="#2B3A55" stroke-width="2.5"/>
  <circle cx="87" cy="74" r="3" fill="#2B3A55"/>
  <circle cx="113" cy="74" r="3" fill="#2B3A55"/>
  <path d="M92 90 Q100 97 108 90" stroke="#C97B5A" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <ellipse cx="79" cy="85" rx="4.5" ry="2.8" fill="#FFB3A7" opacity=".8"/>
  <ellipse cx="121" cy="85" rx="4.5" ry="2.8" fill="#FFB3A7" opacity=".8"/>
</svg>"""

CUSTOM_CSS = """<style>
.block-container { padding-top: 1.4rem; max-width: 900px; }

/* ── 黑板首页 ── */
.board { background: linear-gradient(160deg, #33604F 0%, #24463B 100%);
        border: 10px solid #8B5E3C; border-radius: 16px;
        padding: 26px 30px; color: #F7F3E8;
        box-shadow: 0 10px 26px rgba(36,70,59,.35), inset 0 0 70px rgba(0,0,0,.25); }
.board-inner { display: flex; gap: 28px; align-items: center; }
.board-char { width: 168px; flex-shrink: 0; filter: drop-shadow(0 5px 10px rgba(0,0,0,.35)); }
.board-char svg { width: 100%; height: auto; display: block; }
.board-hello { font-family: "KaiTi","STKaiti","楷体",serif;
        font-size: 1.75rem; font-weight: 700; letter-spacing: 2px; }
.board-msg { font-family: "KaiTi","STKaiti","楷体",serif;
        font-size: 1.1rem; line-height: 1.8; margin-top: 8px; color: #EFE9D8; }
.board-msg strong { color: #FFD98E; }
.board-msg em { color: #FFD98E; font-style: normal; border-bottom: 2px dashed #FFD98E; }
.board-tags { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.board-tags span { font-size: .85rem; border: 1.5px dashed rgba(255,255,255,.55);
        border-radius: 999px; padding: 3px 13px; color: #F1EBDB; }

/* ── 特性小条 ── */
.feature-strip { display: flex; justify-content: center; gap: 26px; flex-wrap: wrap;
        margin: 14px 0 4px; font-size: .9rem; color: #6B7280; }

/* ── 聊天气泡 ── */
[data-testid="stChatMessage"] { border-radius: 14px; padding: 12px 16px;
        background: #FFFFFF; border: 1px solid #E9E4D8;
        box-shadow: 0 1px 6px rgba(90,70,52,.08); }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #FFF6DE; border: 1.5px dashed #E4C878; }

/* ── 侧边栏 ── */
[data-testid="stSidebar"] { background: #FBF7EF; }
.side-mascot { display: flex; align-items: center; gap: 12px; background: #fff;
        border-radius: 16px; padding: 12px; border: 1px solid #EADFC8; margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(90,70,52,.06); }
.side-mascot svg { width: 54px; height: 54px; flex-shrink: 0; }
.side-mascot .t { font-weight: 700; font-size: .98rem; color: #2F5D50; }
.side-mascot .s { font-size: .8rem; color: #8A8171; }

/* ── 按钮主题 ── */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #2F5D50, #3C7A66); border: none; color: #fff; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { filter: brightness(1.12); }
footer { visibility: hidden; }
</style>"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── 黑板首页:苏老师进入系统即打招呼 ──
st.markdown(
    f"""
<div class="board">
  <div class="board-inner">
    <div class="board-char">{TEACHER_SVG}</div>
    <div>
      <div class="board-hello">同学,你好呀 👋</div>
      <div class="board-msg">我是<strong>苏老师</strong>,教了十几年《计算机网络》。你的课件我已经通读完毕,哪里是重点、哪里爱出题,我一清二楚。尽管问——<em>答不上来算我输</em>,但说好了,答完要回去翻课件复习哦 😊</div>
      <div class="board-tags"><span>耐心指数 ★★★★★</span><span>幽默感 ●●●●○</span><span>划重点 从不手软</span><span>口头禅:「翻课件!」</span></div>
    </div>
  </div>
</div>
<div class="feature-strip"><span>🔒 课件本地建库</span><span>📎 句句有出处</span><span>🚫 资料外不瞎编</span></div>
""",
    unsafe_allow_html=True,
)

SUPPORTED = {".pdf", ".txt", ".md"}

SUGGESTIONS = [
    ("📖 什么是三次握手?", "什么是三次握手?为什么需要三次而不是两次?"),
    ("🔍 TCP 和 UDP 怎么选?", "TCP 和 UDP 有什么区别?"),
    ("✏️ 交换机自学习", "以太网交换机是如何自学习的?"),
]


def list_docs() -> list[str]:
    if not config.DOCS_DIR.exists():
        return []
    return sorted(p.name for p in config.DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED)


def has_vectorstore() -> bool:
    return (config.VECTOR_DIR / "index.faiss").exists()


@st.cache_resource(show_spinner="苏老师正在翻开课件…")
def get_chain():
    return build_rag_chain()


def ask(question: str):
    """处理一次提问:展示双方气泡并写入历史。"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="👩‍🏫"):
        try:
            with st.spinner("苏老师正在翻课件找答案…"):
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


# ── 侧边栏:苏老师的资料柜 ────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""<div class="side-mascot">{TEACHER_SVG}
        <div><div class="t">苏老师的资料柜</div><div class="s">喂我课件,才有力气回答</div></div></div>""",
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
    # 苏老师主动发问,以对话开启教学
    st.markdown("**苏老师给你留了三张问题卡,点一张,马上开讲 👇**")
    cols = st.columns(len(SUGGESTIONS))
    for col, (label, full_question) in zip(cols, SUGGESTIONS):
        if col.button(label, use_container_width=True):
            ask(full_question)
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "👩‍🏫"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("📎 引用来源:" + "、".join(msg["sources"]))

question = st.chat_input("同学,今天想问苏老师点什么?")
if question:
    ask(question)
