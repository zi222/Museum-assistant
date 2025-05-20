from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from model_to_llm import model_to_llm
from langchain_core.runnables import RunnableLambda
from get_vector import get_vectordb


vectordb = get_vectordb("./data_base/knowledge_db/national_treasure.pdf","./data_base/vector_db/chroma")

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#问题先被retriever检索得到检索结果，再被combiner进一步处理并输出
combiner = RunnableLambda(combine_docs)
retrieval_chain = retriever | combiner


# OPENAI API 访问密钥配置
GENSTUDIO_API_KEY = "sk-bvzk5vkwe4jxejv2"
DEFAULT_BASE_URL = "https://cloud.infini-ai.com/maas/v1/"

template = """你是一个文物助手，帮助在游览博物馆的游客以引人入胜的方式了解文物的历史背景，使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。尽量使答案简明扼要。
{context}
问题: {input}
"""
# 将template通过 PromptTemplate 转为可以在LCEL中使用的类型
prompt = PromptTemplate(template=template)
llm = model_to_llm(model="deepseek-r1", temperature=0.0, API_KEY=GENSTUDIO_API_KEY, DEFAULT_BASE_URL=DEFAULT_BASE_URL)
qa_chain = (
    RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

question_1 = "什么是司马戊鼎？"
question_2 = "Prompt Engineering for Developer是谁写的？"
result = qa_chain.invoke(question_1)
print("大模型+知识库后回答 question_1 的结果：")
print(result)

# #大模型自己的回答
# print(llm.invoke(question_1).content)

