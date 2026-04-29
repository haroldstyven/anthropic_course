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

def chat(messages, system=None, temperature=0.5):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message.content[0].text

messages = []

"""
# Conversation multiturno
while True:
    user_input = input("> ")
    print(">", user_input)

    add_usser_message(messages, user_input)

    answer = chat(messages)

    add_assistant_message(messages, answer)

    print("---")
    print(answer)
    print("---")
"""

# System prompt
system = """
You are a patient math tutor.
Do not directly answer a student's questions.
Guide them to a solution step by step.
"""

"""
# With system prompt
add_usser_message(messages, "How do I solved 5x+3=2 for x?")
answer = chat(messages, system)
print(answer)

# Without system prompt
add_usser_message(messages, "How do I solved 5x+3=2 for x?")
answer = chat(messages)
print(answer)
"""

"""
# Temperature
add_usser_message(messages, "Generate a idea for a movie.")
answer = chat(messages, temperature=0.0)
print(answer)
"""

# Streaming
add_usser_message(messages, "Writing a 1 sentence description of a fake database.")

with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=messages,
    temperature=0.5,
) as stream:
    for text in stream.text_stream:
        #print(text, end="")
        pass

stream.get_response_text()