from groq import Groq

client = Groq(
    api_key="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",
)

chat_completion = client.chat.completions.create(
    messages=[
        #{"role": "system", "content": ""},
        {
            "role": "user",
            "content": "縱浪大化中 不喜亦不懼 解釋" + "用繁體中文回答",
        }
    ],
    model="mixtral-8x7b-32768",
    temperature=0.9,
    max_tokens=2048,
    top_p=1,
    #stop=None,
    stream=False, # If set, partial message deltas will be sent.
)

print(chat_completion.choices[0].message.content)