from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()
model = "claude-sonnet-4-0"

def add_usser_message(message, text):
    user_message = {
        "role": "user",
        "content": text,
    }
    message.append(user_message)

def add_assistant_message(message, text):
    assistant_message = {
        "role": "assistant",
        "content": text,
    }
    message.append(assistant_message)

def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages
    )
    return message.content[0].text

messages = []

add_usser_message(messages, "What is the capital of Colombia?")

answer = chat(messages)

add_assistant_message(messages, answer)

add_usser_message(messages, "Write another sentence")

answer = chat(messages)


print(messages)