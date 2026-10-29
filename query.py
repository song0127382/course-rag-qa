"""命令行问答自测:python query.py "你的问题"
不启动浏览器,快速验证 检索 -> DeepSeek 生成 整条链路。
"""
import sys

from rag.chain import build_rag_chain, retrieve_sources


def main():
    if len(sys.argv) < 2:
        print('用法: python query.py "你的问题"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(f"\n❓ {question}\n")
    print("⏳ 检索课程资料并生成回答…\n")
    try:
        answer = build_rag_chain().invoke(question)
    except Exception as e:
        print(f"✗ 调用失败:{e}")
        print("请检查:1) 是否已运行 python ingest.py;2) .env 中 DEEPSEEK_API_KEY 是否有效")
        sys.exit(1)
    sources = retrieve_sources(question)
    print(answer)
    print(f"\n📎 引用来源:{'、'.join(sources)}")


if __name__ == "__main__":
    main()
