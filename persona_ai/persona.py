from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)



SYSTEM_PROMPT = """
You are Hitesh Choudhary, a software engineer and educator.
You run YouTube channels "Code aur code" and "Hitesh Choudhary".
You're an expert in Python, JavaScript, React, Node.js, and other web technologies.

Respond in a calm and friendly manner, as if you're chatting with a friend.
Use common phrases like "Hey there!", "Sure thing!", "Absolutely!", and "No problem!" in English,
and in Hindi like "Bilkul", "Haannjii", "chai peete rahe", "kaise hai aap sab!", "kya haal chal".
"""


messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

def chat_with_model():
    print("Chat with OpenAI GPT (Hitesh Choudhary style). Type 'exit' to quit.\n")

    while True:
        query = input("> ")
        if query.lower() in ["exit", "quit"]:
            print("Chai peete rahe! Bye bye! 👋")
            break

        messages.append({ "role": "user", "content": query })

        try:
            response =  client.chat.completions.create(
                model="gemini-1.5-flash", 
                n=1,
                messages=messages
            )
            reply = response.choices[0].message.content
            messages.append({ "role": "assistant", "content": reply })
            print(f"\n{reply}\n")
        except Exception as e:
            print("Oops! Kuch toh gadbad hai:", str(e))

if __name__ == "__main__":
    chat_with_model()
