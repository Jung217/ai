#import os

from groq import Groq

client = Groq(
    api_key="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",
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "用繁體中文回答"},#Translate all answer to Chinese You are a helpful assistant. answer with 繁體中文
        {
            "role": "user",
            "content": "縱浪大化中 不喜亦不懼 解釋",
        }
    ],
    model="llama3-8b-8192",
    temperature=0.8,
)

print(chat_completion.choices[0].message.content)