#import os

from groq import Groq

client = Groq(
    api_key="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",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "安排一套台灣的五日，遊用繁體中文回答",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)