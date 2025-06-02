import os
from dotenv import load_dotenv
from langchain_community.llms import Tongyi

load_dotenv('key.env')  # 指定加载 env 文件
key = os.getenv('DASHSCOPE_API_KEY')  # 获得指定环境变量
DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]  # 获得指定环境变量
model = Tongyi(temperature=1)

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# 加载 arXiv 上的论文《ReAct: Synergizing Reasoning and Acting in Language Models》
loader = ArxivLoader(query="2210.03629", load_max_docs=1)
docs = loader.load()

# 把文本分割成 200 字一组的切片
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name='bge-small-zh-v1.5')

# 构建 FAISS 向量存储和对应的 retriever
vs = FAISS.from_documents(chunks[:10], embedding)
# vs.similarity_search("What is ReAct")
retriever = vs.as_retriever()

# 构建 Document 转文本段落的工具函数
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


# 准备 Model I/O 三元组
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建 RAG 链
chain = (
        {
            "context": retriever | _combine_documents,
            "question": RunnablePassthrough()  # 直接作为 question 的值，不做任何操作
        }
        | prompt
        | model
        | StrOutputParser()
)
print(chain.invoke("什么是 ReAct？"))