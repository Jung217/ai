import os
import openai
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

openai.api_key = os.environ["OPENAI_API_KEY"]

user_input = input("Enter your question: ")

response1 = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
)
print(response1['choices'][0]['message']['content'])