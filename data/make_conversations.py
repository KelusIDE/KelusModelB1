"""Original, deterministic curriculum for B1: paraphrases and real chat turns.

This is synthetic practice data, not broad pretraining. Validation holds out
wording and conversation combinations; it still shares the training topics.
No validation prompt is inserted into a training history or BPE vocabulary.
"""
import json
from pathlib import Path
import random


def build():
    rng = random.Random(104)
    train, validation = [], []

    def add(prompts, answer, heldout):
        train.extend((prompt, answer) for prompt in prompts)
        validation.extend((prompt, answer) for prompt in heldout)

    add(["hello", "hello!", "hi", "hi!", "hey", "hey!", "yo", "yo gng", "hello there", "hey there", "hi Kelus", "hello Kelus", "good morning", "good evening", "hey bro", "sup", "what's up", "hiya", "heyyy", "hiii"],
        "Hi! I'm Kelus Model B1. What would you like to build?", ["hello there!", "hey Kelus!", "hi there, Kelus", "good afternoon"])
    add(["who are you", "who are you?", "what is your name", "what's your name?", "tell me your name", "introduce yourself", "what model are you", "are you Kelus", "who made this model"],
        "I'm Kelus Model B1, a small model trained locally from scratch for basic chat and code.", ["can you introduce yourself?", "what should I call you?"])
    add(["how are you", "how are you?", "how's it going", "how are you doing", "you good", "are you ready"],
        "I'm ready to help. What are you working on?", ["how are things going?", "how are you today?"])
    add(["thanks", "thank you", "thanks!", "ty", "thank you so much", "nice thanks", "that helped", "cool", "nice", "awesome"],
        "You're welcome! What would you like to try next?", ["thanks a lot!", "that's helpful, thank you"])
    add(["bye", "goodbye", "see you", "see ya", "later", "good night"], "See you! Happy coding.", ["bye for now", "talk to you later"])
    add(["what", "what?", "huh", "huh?", "???", "I don't understand", "that makes no sense", "what do you mean", "explain that", "you are speaking gibberish", "that's gibberish", "can you explain", "what was that"],
        "I may have misunderstood. What would you like me to explain?", ["wait, what?", "I didn't get that", "could you clarify?"])
    add(["help", "help me", "can you help", "can you help me", "I need help", "help me code", "can you write code", "let's build something", "I want to make an app"],
        "Sure. Tell me what you want to build and which programming language you want to use.", ["could you help me with coding?", "I'd like to build a program"])
    add(["what can you do", "what can you do?", "what are your abilities", "can you chat", "what do you know", "are you smart", "are you a coding assistant"],
        "I can practice basic chat, explain coding concepts, and write small Python or JavaScript examples. I'm experimental and can make mistakes.", ["what could you help me with?", "what kinds of things can you do?"])
    add(["are you trained from scratch", "are you from scratch", "how were you trained", "do you use a framework", "what framework do you use", "do you use pytorch", "do you use an API", "are you using chatgpt", "are you a pretrained model"],
        "My weights were initialized randomly and trained on local examples. My transformer and gradients are written here; NumPy handles array math.", ["did you start with pretrained weights?", "how does your training work?"])
    add(["can I trust your code", "is your code correct", "are you always right", "will this code work", "can you make mistakes", "is this reliable"],
        "I can make mistakes. Review the code and test it before using it.", ["should I test your code?", "can your answers be wrong?"])
    add(["can you run code", "run this code", "execute my code", "can you open files", "can you browse the internet", "search the web", "can you edit my project"],
        "I generate text only. I cannot run code, browse the web, or edit your files.", ["could you execute a program?", "can you access my computer?"])
    add(["my code is broken", "my code does not work", "my code doesn't work", "it crashes", "there is an error", "help me debug", "fix my code", "it isn't working", "why is my app crashing"],
        "Share the code, the exact error message, and what you expected to happen. Then we can look for the problem.", ["my program keeps crashing", "can you help fix this error?"])
    add(["how do I debug", "how to debug code", "debugging tips", "how do I find a bug", "how should I fix a bug"],
        "Reproduce the problem, read the error message, and test one small change at a time.", ["what is a good way to debug?", "how can I find the problem in my code?"])
    add(["I want to learn coding", "teach me coding", "how do I start coding", "I am new to programming", "I'm a beginner", "how to learn programming", "teach me python"],
        "Start with variables, conditions, loops, and functions. Practice by writing small programs and testing them.", ["where should a programming beginner start?", "help me learn to program"])
    add(["tell me a joke", "say something funny", "coding joke", "make me laugh", "tell a programming joke"],
        "Why did the programmer bring a ladder? To reach a higher level!", ["do you know a programming joke?", "tell me a funny joke"])
    add(["what is Minecraft", "do you know Minecraft", "tell me about Minecraft", "what's minecraft"],
        "Minecraft is a sandbox game where players build, explore, and survive in a world of blocks.", ["can you explain Minecraft?", "what kind of game is Minecraft?"])
    add(["what is Lunar client", "what is Badlion client", "how do I get more Minecraft fps", "improve Minecraft fps", "Minecraft is lagging", "how to optimize minecraft"],
        "For Minecraft performance, try lower render distance, fewer particles, and lighter graphics settings. Measure FPS before and after each change.", ["how can I improve my Minecraft performance?", "my Minecraft fps is low"])
    add(["make a complete game", "build a huge app", "write an operating system", "train a large AI", "explain quantum physics", "what is the weather", "what is the latest news", "asdfgh", "qwerty", "zxqv", "do my homework", "tell me everything", "something else"],
        "That is outside what I can reliably answer. Could you ask a small coding question or describe one specific part?", ["build me a whole operating system", "what happened in the news today?"])

    concepts = [
        ("variable", "A variable is a name that refers to a value.", "x = 5"),
        ("function", "A function is a reusable block of code that takes inputs and can return a result.", "def double(x):\n    return x * 2"),
        ("loop", "A loop repeats a block of code.", "for i in range(3):\n    print(i)"),
        ("list", "A list stores an ordered collection of values.", "items = [1, 2, 3]"),
        ("dictionary", "A dictionary maps keys to values.", 'person = {"name": "Sam"}'),
        ("boolean", "A boolean is a value that is either True or False.", "ready = True"),
        ("string", "A string is text enclosed in quotes.", 'name = "Sam"'),
        ("integer", "An integer is a whole number without a fractional part.", "count = 10"),
        ("condition", "A condition is an expression that evaluates to True or False.", "if score > 10:\n    print(\"You win!\")"),
        ("return statement", "A return statement sends a result back from a function.", "def add(a, b):\n    return a + b"),
        ("syntax error", "A syntax error means the code does not follow the language's grammar.", 'print("Hello")'),
        ("test", "A test checks that code behaves as expected for a given input.", "assert 2 + 3 == 5"),
        ("comment", "A comment explains code and is ignored when the program runs.", "# This is a comment"),
        ("recursion", "Recursion is when a function calls itself. It needs a base case to stop.", "def countdown(n):\n    if n > 0:\n        countdown(n - 1)"),
        ("algorithm", "An algorithm is a sequence of steps for solving a problem.", "numbers = [3, 1, 2]\nordered = sorted(numbers)"),
        ("exception", "An exception is an error that interrupts normal program execution.", "raise ValueError(\"Invalid value\")"),
        ("Python", "Python is a programming language used for scripts, apps, and data work.", 'print("Hello, world!")'),
        ("JavaScript", "JavaScript is a programming language often used to add behavior to web pages.", 'console.log("Hello, world!");'),
        ("HTML", "HTML describes the structure of a web page.", "<h1>Hello</h1>"),
        ("CSS", "CSS controls how a web page looks.", "body { color: green; }"),
    ]
    for topic, explanation, _ in concepts:
        add([f"what is a {topic}", f"what is {topic}", f"what is a {topic}?", f"explain {topic}", f"explain a {topic}", f"tell me about {topic}", f"define {topic}", f"what does {topic} mean", f"teach me about {topic}", f"what's a {topic}"],
            explanation, [f"can you explain {topic}?", f"what does a {topic} do?"])

    snippets = [
        ("add", ["add two numbers", "addition function", "add function", "function to add two numbers"], "def add(a, b):\n    return a + b"),
        ("subtract", ["subtract two numbers", "subtraction function", "subtract function"], "def subtract(a, b):\n    return a - b"),
        ("multiply", ["multiply two numbers", "multiplication function", "multiply function"], "def multiply(a, b):\n    return a * b"),
        ("square", ["square a number", "square function", "function that squares a number"], "def square(x):\n    return x * x"),
        ("double", ["double a number", "double function", "function to double a number"], "def double(x):\n    return x * 2"),
        ("is_even", ["check if a number is even", "even number function", "is_even function"], "def is_even(n):\n    return n % 2 == 0"),
        ("is_odd", ["check if a number is odd", "odd number function", "is_odd function"], "def is_odd(n):\n    return n % 2 != 0"),
        ("reverse", ["reverse a string", "string reverse function", "reverse function"], "def reverse(text):\n    return text[::-1]"),
        ("length", ["get the length of a list", "list length function", "length function"], "def length(items):\n    return len(items)"),
        ("total", ["sum a list", "list sum function", "total function"], "def total(values):\n    return sum(values)"),
        ("greet", ["greet someone", "greeting function", "greet function"], 'def greet(name):\n    return f"Hello, {name}!"'),
        ("hello", ["hello world", "print hello world", "hello world program"], 'print("Hello, world!")'),
        ("sort", ["sort a list", "sort numbers", "sorting example"], "numbers = [3, 1, 2]\nordered = sorted(numbers)"),
        ("loop", ["for loop", "loop example", "print numbers from 0 to 4"], "for i in range(5):\n    print(i)"),
    ]
    code_pairs = []
    for name, descriptions, code in snippets:
        answer = "```python\n" + code + "\n```"
        prompts = []
        for description in descriptions:
            prompts.extend([f"write a Python {description}", f"{description} in Python", f"show me how to {description} in Python", f"write Python code to {description}", f"can you {description} in Python", f"make a Python {description}", f"I need a Python {description}", f"please write a Python {description}"])
        heldout = [f"could you show a Python example to {descriptions[0]}?", f"help me write Python to {descriptions[0]}"]
        add(prompts, answer, heldout)
        code_pairs.append((prompts, answer, heldout))

    js_examples = [
        ("add function", "function add(a, b) {\n  return a + b;\n}"),
        ("square function", "function square(x) {\n  return x * x;\n}"),
        ("hello world", 'console.log("Hello, world!");'),
        ("double function", "function double(x) {\n  return x * 2;\n}"),
    ]
    for description, code in js_examples:
        add([f"write a JavaScript {description}", f"{description} in JavaScript", f"make a JavaScript {description}", f"show me a JavaScript {description}", f"I need a JavaScript {description}"],
            "```javascript\n" + code + "\n```", [f"please give me a JavaScript {description} example"])

    # Case/punctuation augmentation is confined to the training split.
    base_train = list(train)
    for prompt, answer in base_train:
        for variant in (prompt[0].upper() + prompt[1:], prompt + ".", prompt + "!", prompt.upper()):
            train.append((variant, answer))
    train = list(dict.fromkeys(train))
    validation = list(dict.fromkeys(validation))
    heldout_text = {prompt for prompt, _ in validation}
    train = [(prompt, answer) for prompt, answer in train if prompt not in heldout_text]

    def messages(pairs):
        return [message for prompt, answer in pairs for message in ({"role": "user", "content": prompt}, {"role": "assistant", "content": answer})]

    records = [{"split": "train", "messages": messages([pair])} for pair in train]
    records += [{"split": "validation", "messages": messages([pair])} for pair in validation]
    # Mix chat and code in the same history, including greeting after code.
    for _ in range(2000):
        records.append({"split": "train", "messages": messages(rng.sample(train, rng.choice((2, 2, 3, 4))))})
    for _ in range(50):
        records.append({"split": "validation", "messages": messages(rng.sample(validation, 3))})
    # Grounded follow-ups: the example depends on the preceding concept.
    for topic, explanation, code in concepts:
        language = "javascript" if topic == "JavaScript" else "html" if topic == "HTML" else "css" if topic == "CSS" else "python"
        example = f"```{language}\n{code}\n```"
        for first in (f"what is a {topic}", f"explain {topic}", f"tell me about {topic}"):
            for follow in ("show me an example", "give me an example", "example please", "can you show an example", "how does that look in code"):
                records.append({"split": "train", "messages": messages([(first, explanation), (follow, example)])})
        records.append({"split": "validation", "messages": messages([(f"can you explain {topic}?", explanation), ("could you give an example?", example)])})
    # A few multi-turn chains of the bug reports, not held-out generalization.
    for _ in range(150):
        chosen = [rng.choice(code_pairs)[0][0], rng.choice(["hello!", "hi!", "hello", "what", "huh?"])]
        mapping = dict(train)
        records.append({"split": "train", "messages": messages([(p, mapping[p]) for p in chosen])})
    seen, unique = set(), []
    for record in records:
        key = json.dumps(record["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key); unique.append(record)
    return unique


if __name__ == "__main__":
    records = build()
    destination = Path(__file__).with_name("conversations.jsonl")
    destination.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    counts = {split: sum(record["split"] == split for record in records) for split in ("train", "validation")}
    print(json.dumps({"path": str(destination), "counts": counts}))
