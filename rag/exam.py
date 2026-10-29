"""期末考点预测:逐章分析课件 -> 汇总生成考试重点与题型预测报告。

流程:
1. 从 FAISS 知识库取回全部文本块,按课件(章)分组;
2. 每份课件调用一次模型做考点分析(核心考点、题型、概率、预测例题);
3. 汇总各章分析,生成总报告(优先级总览/重点报告/题型预测/临考叮嘱)。
"""
from langchain_deepseek import ChatDeepSeek
import config
from rag.vectorstore import load_vectorstore

MAX_CHARS_PER_SOURCE = 12000   # 单份课件送入模型的最大字符数
MAX_ANALYSIS_CHARS = 2500      # 单章分析结果保留的最大字符数

COMMON_TYPES = ["选择题", "填空题", "判断题", "名词解释", "简答题", "计算题", "综合应用题"]

CHAPTER_PROMPT = """你是经验丰富的大学老师"苏老师"。下面是课程课件《{source}》的全部内容。
学生告诉我,他们学校这门课期末考试的题型有:{types}。{extra}

请只依据上面的课件内容,输出 Markdown(不要输出课件里没有的知识点):

### 《{source}》考点分析

按重要性列出 3~6 个核心考点,每个考点按以下格式:

**考点X:名称**(出题概率:★★★★☆ | 题型:简答题、计算题)
- 内容概述:1~2 句话讲清这个考点
- 预测例题:一道最可能出的题目
- 答题要点:2~3 个关键得分点

最后给一行"本章复习建议"。

课件内容:
{content}"""

FINAL_PROMPT = """你是"苏老师"。你已经逐章分析了学生的课件,各章分析结果如下:

{analyses}

学生告诉我,他们学校这门课期末考试的题型有:{types}。{extra}

请汇总生成一份完整的 Markdown 报告,严格遵守以下结构:

# 📝 期末考试重点与题型预测报告

## 一、复习优先级总览
用 Markdown 表格列出每份课件:章节(课件名)| 重点程度(★) | 主考题型 | 建议复习时间占比

## 二、考试重点报告
按重点程度从高到低,逐章列出必须掌握的知识点,精炼、条理清晰。

## 三、题型预测报告
按题型归类:每种题型最可能出哪些章节、哪些知识点、答题时要注意什么。

## 四、苏老师的临考叮嘱
3~5 条接地气的复习建议(结合上面分析,比如"计算题集中在第5章,把刷题时间花在这里")。

要求:只依据各章分析结果汇总,不要编造课件中没有的知识点。"""


def _llm() -> ChatDeepSeek:
    # temperature 调低:考点分析需要克制、贴原文,不要发散
    return ChatDeepSeek(model=config.CHAT_MODEL, temperature=0.3)


def group_chunks_by_source() -> dict:
    """从向量库取回全部文本块,按课件(来源文件)合并分组。"""
    vectorstore = load_vectorstore()
    grouped: dict = {}
    for doc in vectorstore.docstore._dict.values():
        source = doc.metadata.get("source", "未知课件")
        grouped.setdefault(source, []).append(doc.page_content)
    return {
        source: "\n\n".join(parts)[:MAX_CHARS_PER_SOURCE]
        for source, parts in sorted(grouped.items())
    }


def run_prediction(types: list, extra: str = "", progress_cb=None) -> str:
    """逐章分析并汇总报告。progress_cb(str) 用于向界面汇报进度。"""
    if not types:
        raise ValueError("请至少选择一种考试题型")
    extra_note = f"\n学生补充说明:{extra}" if extra.strip() else ""
    llm = _llm()
    groups = group_chunks_by_source()

    analyses = []
    for i, (source, content) in enumerate(groups.items(), 1):
        if progress_cb:
            progress_cb(f"📖 正在分析第 {i}/{len(groups)} 份课件:{source}")
        prompt = CHAPTER_PROMPT.format(
            source=source, types="、".join(types), extra=extra_note, content=content
        )
        answer = llm.invoke(prompt).content
        # 兼容新旧 langchain 返回类型
        if not isinstance(answer, str):
            answer = "".join(part.text for part in answer)
        analyses.append(answer[:MAX_ANALYSIS_CHARS])

    if progress_cb:
        progress_cb("🧮 各章分析完毕,正在汇总生成报告…")
    final_prompt = FINAL_PROMPT.format(
        analyses="\n\n".join(analyses), types="、".join(types), extra=extra_note
    )
    report = llm.invoke(final_prompt).content
    if not isinstance(report, str):
        report = "".join(part.text for part in report)

    # 把逐章详细分析附在报告末尾,保住预测例题等细节
    detail = "\n\n---\n\n## 五、各章详细考点与预测题(逐章分析)\n\n" + "\n\n".join(analyses)
    return report + detail
