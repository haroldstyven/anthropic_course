from dotenv import load_dotenv
from anthropic import Anthropic
import json
from statistics import mean
import ast
import re

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
"""

prompt = """
Generate three differents sample AWS CLI commands. Each should be very shot.
"""

"""
# Structured data - controlling output

add_usser_message(messages, "Generate a very short event bridge rule as json")
add_assistant_message(messages, "```json")

text = chat(messages, stop_sequenc=["```"])

print(text)

rule = json.loads(text.strip())
print(rule)

# other example

add_usser_message(messages, prompt)
add_assistant_message(messages, "Here are all three commands in a single block without any comments:\n ```bash")

text = chat(messages, stop_sequence=["```"])
print(text.strip())
"""

# Prompt evaluation

def generate_dataset():
    prompt = """
    Generate a evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
    that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
    each representing task that requires Python, JSON, or a Regex to complete.

    Example output:
    ```json
    [
        {
            "task": "Description of task",
        },
        ...additional
    ]
    ```

    * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
    * Focus on tasks that do not require writing much code

    Please generate 3 objects.
    """

    add_usser_message(messages, prompt)
    add_assistant_message(messages, "```json")

    text = chat(messages, stop_sequence=["```"])
    return json.loads(text)

"""
dataset = generate_dataset()

with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
"""

def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0

def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0

def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0

def grade_syntax(response, test_case):
    format = test_case["format"]

    if format == "python":
        return validate_python(response)
    elif format == "json":
        return validate_json(response)
    else:
        return validate_regex(response)

# Running the eval
def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
    Please solve the following task:

    {test_case["task"]}

    * Respond only with Python, JSON, or a plain Regex
    * Do not add any comments or commentary or explanation
    """
    
    messages = []
    add_usser_message(messages, prompt)
    add_assistant_message(messages, "```code")

    output = chat(messages, stop_sequence=["```"])
    return output

def grade_by_model(test_case, output):
    # Create evaluation prompt
    eval_prompt = """
    You are an expert code reviewer. Evaluate this AI-generated solution.
    
    Task: {task}
    Solution: {solution}
    
    Provide your evaluation as a structured JSON object with:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement  
    - "reasoning": A concise explanation of your assessment
    - "score": A number between 1-10
    """
    
    messages = []
    add_usser_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")
    
    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)

def run_test_case(test_case):
    output = run_prompt(test_case)
    
    # Grade the output
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    syntax_score = grade_syntax(output, test_case)

    score = (model_score + syntax_score) / 2
    
    return {
        "output": output, 
        "test_case": test_case, 
        "score": score,
        "reasoning": reasoning
    }

def run_eval(dataset):
    results = []
    
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    
    average_score = mean([result["score"] for result in results])
    print(f"Average score: {average_score}")
    
    return results

with open("dataset.json", "r") as f:
    dataset = json.load(f)

results = run_eval(dataset)

print(json.dumps(results, indent=2))