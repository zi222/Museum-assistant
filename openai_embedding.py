from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GENSTUDIO_API_KEY")
DEFAULT_BASE_URL = os.getenv("DEFAULT_BASE_URL")


from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List

# Define a callback handler to process streaming tokens
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)

# Initialize the ChatOpenAI model with streaming enabled
llm_streaming = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base=DEFAULT_BASE_URL,
    model="deepseek-r1",
    streaming=False,
    temperature=0.0,
    callbacks=[StreamHandler()]  # Pass the callback handler
)

# Define your messages
messages = [
    {"role": "system", "content": "你是一个文物小助手."},
    {"role": "user", "content": "四羊方尊是什么年代的?"}
]

# Get a response from the chat model
response = llm_streaming.invoke(input=messages)

# Output the response (optional, as it is already printed by the callback)
print("\n\n***********\n\n以下是 LLM 的完整回复:\n\n", response)