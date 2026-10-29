"""RAG 问答链:检索相关片段 -> 拼入提示词 -> DeepSeek 生成回答。"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_deepseek import ChatDeepSeek

import config

PROMPT_TEMPLATE = """你是一名耐心的课程助教,请只依据下面提供的课程资料回答学生的问题。

要求:
1. 回答要准确、条理清晰,使用中文。
2. 只使用资料中的内容作答;资料中没有涉及的内容,明确说明"课程资料中未涉及",不要编造。
3. 回答末尾用一行列出引用的资料来源,格式如:【来源】课件名

课程资料:
{context}

学生的问题:
{question}"""


def format_docs(docs):
    """把检索到的文本块拼成提示词里的 context,并保留来源标记。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[片段{i} | 来源:{source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain():
    """构建 RAG 链(LCEL):检索 -> 格式化 -> 提示词 -> DeepSeek -> 文本输出。"""
    from rag.vectorstore import get_retriever

    retriever = get_retriever()
    # ChatDeepSeek 自动读取环境变量 DEEPSEEK_API_KEY(来自 .env)
    llm = ChatDeepSeek(model=config.CHAT_MODEL, temperature=0.7)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def retrieve_sources(question: str):
    """单独检索相关片段,用于在界面上展示引用来源。"""
    from rag.vectorstore import get_retriever

    docs = get_retriever().invoke(question)
    return list({doc.metadata.get("source", "未知来源") for doc in docs})
