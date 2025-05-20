from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from get_vector import get_vectordb
from model_to_llm import model_to_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# OPENAI API 访问密钥配置
GENSTUDIO_API_KEY = "sk-bvzk5vkwe4jxejv2"
DEFAULT_BASE_URL = "https://cloud.infini-ai.com/maas/v1/"
llm = model_to_llm(model="deepseek-r1", temperature=0.0, API_KEY=GENSTUDIO_API_KEY, DEFAULT_BASE_URL=DEFAULT_BASE_URL)

vectordb = get_vectordb("./data_base/knowledge_db/national_treasure.pdf","./data_base/vector_db/chroma")
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 压缩问题的系统 prompt
condense_question_system_template = (
    "请根据聊天记录完善用户最新的问题，"
    "如果用户最新的问题不需要完善则返回用户的问题。"
    )
# 构造 压缩问题的 prompt template
condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
# 构造检索文档的链
# RunnableBranch 会根据条件选择要运行的分支
retrieve_docs = RunnableBranch(
    # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
    (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
    # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
    condense_question_prompt | llm | StrOutputParser() | retriever,
)


# 问答链的系统prompt
system_prompt = (
    "你是一个文物问答的助手。 "
    "请使用检索到的上下文片段回答这个问题。 "
    "如果你不知道答案就说不知道。 "
    "请使用简洁的话语回答用户。"
    "\n\n"
    "{context}"
)
# 制定prompt template
qa_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
# 重新定义 combine_docs
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"]) # 将 docs 改为 docs["context"]
# 定义问答链
qa_chain = (
    RunnablePassthrough.assign(context=combine_docs) # 使用 combine_docs 函数整合 qa_prompt 中的 context
    | qa_prompt # 问答模板
    | llm
    | StrOutputParser() # 规定输出的格式为 str
)
# 定义带有历史记录的问答链
qa_history_chain = RunnablePassthrough.assign(
    context = (lambda x: x) | retrieve_docs # 将查询结果存为 content
    ).assign(answer=qa_chain) # 将最终结果存为 answer

# 带聊天记录
result_1=qa_history_chain.invoke({
    "input": "司母戊鼎的历史背景是什么？",
    "chat_history": [
        ("human", "介绍一下司母戊鼎"),
        ("ai", "它是商后期（约公元前14-前11世纪）的青铜方鼎，通高133厘米，重832.84公斤，是目前已知中国古代最重的青铜器。1939年出土于河南安阳，现藏中国国家博物馆。"),
    ]
})
print("带聊天记录的问答结果：")
print(result_1)
# 不带聊天记录
result_2=qa_history_chain.invoke({
    "input": "司母戊鼎的历史背景是什么？",
    "chat_history": []
})
print("不带聊天记录的问答结果：")
print(result_2)
