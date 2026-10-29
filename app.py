"""课程知识库问答助手 - 「苏老师」私人助教界面(黑板粉笔风)
运行: streamlit run app.py
"""
import streamlit as st

import config
from rag.chain import build_rag_chain, retrieve_sources
from rag.loader import load_documents, split_documents
from rag.vectorstore import build_vectorstore

st.set_page_config(page_title="苏老师 · 你的私人课程助教", page_icon="🎓", layout="centered")

# ── 人物形象「苏老师」:挥粉笔打招呼、发髻别铅笔、臂弯夹课本(SVG 手绘 + CSS 动画)──
TEACHER_SVG = """<svg viewBox="0 0 250 270" xmlns="http://www.w3.org/2000/svg">
  <!-- 西装身体 -->
  <path d="M62 262 L68 196 Q74 158 112 158 Q150 158 156 196 L162 262 Z" fill="#3E7263"/>
  <path d="M96 162 L112 196 L128 162 Q120 156 112 156 Q104 156 96 162 Z" fill="#F7F2E7"/>
  <path d="M96 162 L112 196 L102 199 L88 170 Z" fill="#33604F"/>
  <path d="M128 162 L112 196 L122 199 L136 170 Z" fill="#33604F"/>
  <circle cx="112" cy="206" r="4.5" fill="#E8B04B"/>
  <circle cx="112" cy="224" r="4" fill="#E8B04B"/>
  <!-- 左臂:臂弯夹课本 -->
  <path d="M76 172 Q60 188 57 208 L74 213 Q79 194 90 184 Z" fill="#33604F"/>
  <circle cx="67" cy="216" r="10" fill="#F8C9A0"/>
  <g transform="rotate(-10 56 206)">
    <rect x="28" y="190" width="54" height="34" rx="4" fill="#D9A03F"/>
    <rect x="32" y="194" width="46" height="26" rx="2" fill="#FDF8EC"/>
    <line x1="55" y1="194" x2="55" y2="220" stroke="#D9C48E" stroke-width="2"/>
    <path d="M38 200 h13 M38 206 h13" stroke="#C9B78C" stroke-width="2" stroke-linecap="round"/>
    <path d="M60 200 h12 M60 206 h12" stroke="#C9B78C" stroke-width="2" stroke-linecap="round"/>
  </g>
  <!-- 右臂:扬起挥手(会动) -->
  <g class="wave-arm">
    <path d="M138 170 Q170 160 188 130 L204 141 Q186 174 152 186 Z" fill="#33604F"/>
    <circle cx="198" cy="124" r="12" fill="#F8C9A0"/>
    <g transform="rotate(-28 198 116)">
      <rect x="193" y="102" width="9" height="20" rx="2.5" fill="#FFFFFF" stroke="#DDD6C6" stroke-width="1.5"/>
    </g>
    <path d="M176 96 q-7 11 -1 22" stroke="rgba(255,255,255,.6)" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <path d="M167 102 q-5 8 -1 16" stroke="rgba(255,255,255,.4)" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  </g>
  <!-- 脖子与后发 -->
  <rect x="104" y="140" width="16" height="18" rx="6" fill="#F8C9A0"/>
  <circle cx="112" cy="96" r="52" fill="#6B4F3A"/>
  <circle cx="66" cy="120" r="14" fill="#6B4F3A"/>
  <circle cx="158" cy="120" r="14" fill="#6B4F3A"/>
  <path d="M158 132 q11 14 3 27" stroke="#6B4F3A" stroke-width="3.5" fill="none" stroke-linecap="round"/>
  <!-- 发髻与铅笔 -->
  <circle cx="112" cy="34" r="18" fill="#6B4F3A"/>
  <rect x="97" y="46" width="30" height="6" rx="3" fill="#E8B04B"/>
  <g transform="rotate(24 132 30)">
    <rect x="118" y="27" width="42" height="6" rx="2" fill="#E8B04B"/>
    <polygon points="160,27 169,30 160,33" fill="#4A3B2F"/>
    <rect x="113" y="27" width="7" height="6" rx="2" fill="#E88B8B"/>
  </g>
  <!-- 脸 -->
  <ellipse cx="112" cy="104" rx="37" ry="35" fill="#F8C9A0"/>
  <path d="M75 96 Q80 62 112 62 Q144 62 149 96 Q132 74 112 77 Q92 74 75 96 Z" fill="#6B4F3A"/>
  <!-- 眉毛与圆框眼镜 -->
  <path d="M85 93 q8 -5 17 -1" stroke="#6B4F3A" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <path d="M122 92 q8 -4 16 1" stroke="#6B4F3A" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <circle cx="94" cy="108" r="11.5" fill="#FFFFFF" fill-opacity=".25" stroke="#4A3B2F" stroke-width="2.6"/>
  <circle cx="130" cy="108" r="11.5" fill="#FFFFFF" fill-opacity=".25" stroke="#4A3B2F" stroke-width="2.6"/>
  <line x1="105.5" y1="108" x2="118.5" y2="108" stroke="#4A3B2F" stroke-width="2.6"/>
  <line x1="82.5" y1="107" x2="76" y2="103" stroke="#4A3B2F" stroke-width="2.6"/>
  <line x1="141.5" y1="107" x2="148" y2="103" stroke="#4A3B2F" stroke-width="2.6"/>
  <!-- 笑成弯月的眼睛 -->
  <path d="M88 110 q6 -7 12 0" stroke="#4A3B2F" stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="M124 110 q6 -7 12 0" stroke="#4A3B2F" stroke-width="3" fill="none" stroke-linecap="round"/>
  <!-- 微笑与腮红 -->
  <path d="M102 127 Q112 137 122 127" stroke="#C97B5A" stroke-width="3" fill="none" stroke-linecap="round"/>
  <ellipse cx="81" cy="121" rx="6" ry="3.5" fill="#F5A08C" opacity=".55"/>
  <ellipse cx="143" cy="121" rx="6" ry="3.5" fill="#F5A08C" opacity=".55"/>
</svg>"""

# ── 黑板角落的粉笔涂鸦:网络拓扑(左上)、地球仪与星星(右下)──
DOODLE_TL = """<svg viewBox="0 0 130 95" xmlns="http://www.w3.org/2000/svg">
  <g stroke="rgba(255,255,255,.42)" stroke-width="2" fill="rgba(255,255,255,.12)">
    <line x1="22" y1="22" x2="62" y2="14"/><line x1="62" y1="14" x2="102" y2="30"/>
    <line x1="22" y1="22" x2="48" y2="62"/><line x1="102" y1="30" x2="88" y2="66"/>
    <line x1="48" y1="62" x2="88" y2="66"/><line x1="62" y1="14" x2="48" y2="62"/>
    <circle cx="22" cy="22" r="8"/><circle cx="62" cy="14" r="8"/><circle cx="102" cy="30" r="8"/>
    <circle cx="48" cy="62" r="8"/><circle cx="88" cy="66" r="8"/>
  </g>
  <path d="M30 21 h18" stroke="rgba(255,255,255,.75)" stroke-width="2.5" stroke-dasharray="4 4"/>
</svg>"""

DOODLE_BR = """<svg viewBox="0 0 150 110" xmlns="http://www.w3.org/2000/svg">
  <g stroke="rgba(255,255,255,.45)" stroke-width="2" fill="none">
    <circle cx="34" cy="74" r="16"/>
    <ellipse cx="34" cy="74" rx="16" ry="6"/>
    <ellipse cx="34" cy="74" rx="6" ry="16"/>
  </g>
  <path class="tw1" d="M84 22 l3 7 7 3 -7 3 -3 7 -3 -7 -7 -3 7 -3 z" fill="rgba(255,255,255,.7)"/>
  <path class="tw2" d="M122 78 l2.5 5.5 5.5 2.5 -5.5 2.5 -2.5 5.5 -2.5 -5.5 -5.5 -2.5 5.5 -2.5 z" fill="rgba(255,255,255,.55)"/>
  <path class="tw3" d="M70 90 l2 4.5 4.5 2 -4.5 2 -2 4.5 -2 -4.5 -4.5 -2 4.5 -2 z" fill="rgba(255,255,255,.5)"/>
</svg>"""

CUSTOM_CSS = """<style>
.block-container { padding-top: 1.4rem; max-width: 900px; }

/* ── 黑板首页 ── */
.board { position: relative; overflow: hidden;
        background: linear-gradient(160deg, #33604F 0%, #24463B 100%);
        border: 10px solid #8B5E3C; border-radius: 16px;
        padding: 26px 30px; color: #F7F3E8;
        box-shadow: 0 10px 26px rgba(36,70,59,.35), inset 0 0 70px rgba(0,0,0,.25); }
.board-doodles { position: absolute; inset: 0; pointer-events: none; }
.doodle-tl { position: absolute; top: 10px; left: 14px; width: 110px; opacity: .8; }
.doodle-br { position: absolute; bottom: 2px; right: 10px; width: 92px; opacity: .55; }
.board-inner { position: relative; display: flex; gap: 26px; align-items: center; }
.board-char { width: 190px; flex-shrink: 0; filter: drop-shadow(0 5px 10px rgba(0,0,0,.35)); }
.board-char svg { width: 100%; height: auto; display: block; animation: bob 3.6s ease-in-out infinite; }
@keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
.wave-arm { transform-box: view-box; transform-origin: 142px 175px; animation: wave 1.8s ease-in-out infinite; }
@keyframes wave { 0%,100% { transform: rotate(7deg); } 50% { transform: rotate(-17deg); } }
.tw1 { animation: twinkle 2.2s ease-in-out infinite .2s; }
.tw2 { animation: twinkle 2.8s ease-in-out infinite 1s; }
.tw3 { animation: twinkle 2.5s ease-in-out infinite 1.6s; }
@keyframes twinkle { 0%,100% { opacity: .25; } 50% { opacity: .9; } }
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

/* ── 侧边栏(静默版形象,不做动画)── */
[data-testid="stSidebar"] { background: #FBF7EF; }
.side-mascot svg { animation: none; }
.side-mascot .wave-arm { animation: none; }
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

# ── 黑板首页:苏老师进入系统即挥手打招呼 ──
st.markdown(
    f"""
<div class="board">
  <div class="board-doodles"><div class="doodle-tl">{DOODLE_TL}</div><div class="doodle-br">{DOODLE_BR}</div></div>
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
