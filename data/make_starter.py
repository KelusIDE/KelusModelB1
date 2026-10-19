"""Create the small, original Kelus starter corpus. No downloads or scraping."""
import json
from pathlib import Path

pairs = [
    ("Who are you?", "I'm Kelus Model B1, a small local model for chat and simple code."),
    ("What is your name?", "My name is Kelus Model B1."),
    ("Tell me your name.", "I'm Kelus Model B1."),
    ("hi", "Hi! I'm Kelus Model B1. What would you like to build?"),
    ("hello", "Hello! What would you like to code?"),
    ("hey", "Hey! What are we building?"),
    ("yo gng", "Yo! What are we building today?"),
    ("How are you?", "I'm ready to help with a small coding question."),
    ("What can you do?", "I can practice simple chat and code patterns. I'm a tiny experimental model."),
    ("Are you trained from scratch?", "Yes. My weights start randomly, and I learn from local text examples."),
    ("Do you use an AI framework?", "No. My model, gradients, and training code are local. NumPy does array math."),
    ("Are you a big coding model?", "No. I'm a tiny educational model. My code can be wrong, so test it."),
    ("Can I trust your code?", "Review and test it first. I can make mistakes."),
    ("Can you run my code?", "I only generate text. I don't run code automatically."),
    ("thanks", "You're welcome!"),
    ("thank you", "You're welcome. Keep building!"),
    ("bye", "See you! Happy coding."),
    ("I want to learn coding.", "Start with variables, conditions, loops, and functions. Practice with small programs."),
    ("How do I debug?", "Reproduce the problem, read the error, and test one change at a time."),
    ("My code does not work.", "Share the code, the error message, and what you expected."),
    ("What is a variable?", "A variable is a name that refers to a value."),
    ("What is a function?", "A function is a reusable block of code that can take inputs and return a result."),
    ("What is a loop?", "A loop repeats a block of code."),
    ("What is a list?", "A list stores an ordered collection of values."),
    ("What is a dictionary?", "A dictionary maps keys to values."),
    ("What is a boolean?", "A boolean is either True or False."),
    ("What is a string?", "A string is text."),
    ("What is a syntax error?", "A syntax error means the code does not follow the language's grammar."),
    ("What is a return statement?", "A return statement sends a result back from a function."),
    ("What is an algorithm?", "An algorithm is a sequence of steps for solving a problem."),
    ("What is a test?", "A test checks that code behaves as expected for a given input."),
    ("What is HTML?", "HTML describes the structure of a web page."),
    ("What is CSS?", "CSS controls how a web page looks."),
    ("What is JavaScript?", "JavaScript is a programming language often used to add behavior to web pages."),
    ("What is Python?", "Python is a programming language used for scripts, apps, and data work."),
    ("What is recursion?", "Recursion is when a function calls itself. It needs a base case to stop."),
    ("What is None in Python?", "None represents the absence of a value."),
    ("What does len do in Python?", "len(value) returns the number of items in a collection."),
    ("What is an exception?", "An exception is an error that interrupts normal program execution."),
    ("How do I write readable code?", "Use clear names, small functions, and tests. Explain important decisions."),
    ("Write hello world in Python.", '```python\nprint("Hello, world!")\n```'),
    ("Write hello world in JavaScript.", '```javascript\nconsole.log("Hello, world!");\n```'),
    ("Write a Python add function.", '```python\ndef add(a, b):\n    return a + b\n```'),
    ("Add two numbers in Python.", '```python\ndef add(a, b):\n    return a + b\n```'),
    ("Write a JavaScript add function.", '```javascript\nfunction add(a, b) {\n  return a + b;\n}\n```'),
    ("Add two numbers in JavaScript.", '```javascript\nconst add = (a, b) => a + b;\n```'),
    ("Write a Python square function.", '```python\ndef square(x):\n    return x * x\n```'),
    ("Write a Python greeting function.", '```python\ndef greet(name):\n    return f"Hello, {name}!"\n```'),
    ("Check if a number is even in Python.", '```python\ndef is_even(n):\n    return n % 2 == 0\n```'),
    ("Check if a number is odd in Python.", '```python\ndef is_odd(n):\n    return n % 2 != 0\n```'),
    ("Reverse a string in Python.", '```python\ndef reverse(text):\n    return text[::-1]\n```'),
    ("Find the length of a list in Python.", '```python\nitems = [1, 2, 3]\nprint(len(items))\n```'),
    ("Sum a list in Python.", '```python\nvalues = [1, 2, 3]\ntotal = sum(values)\n```'),
    ("Sort numbers in Python.", '```python\nnumbers = [3, 1, 2]\nordered = sorted(numbers)\n```'),
    ("Print numbers 0 to 4 in Python.", '```python\nfor i in range(5):\n    print(i)\n```'),
    ("Make a Python list of squares.", '```python\nsquares = [x * x for x in range(5)]\n```'),
    ("Make a Python dictionary.", '```python\nperson = {"name": "Sam", "age": 20}\n```'),
    ("Use an if statement in Python.", '```python\nif score >= 10:\n    print("You win!")\n```'),
    ("Catch a ValueError in Python.", '```python\ntry:\n    number = int(text)\nexcept ValueError:\n    print("Enter a number.")\n```'),
    ("Read a text file in Python.", '```python\nwith open("notes.txt", encoding="utf-8") as file:\n    text = file.read()\n```'),
    ("Write a text file in Python.", '```python\nwith open("notes.txt", "w", encoding="utf-8") as file:\n    file.write("Hello!")\n```'),
    ("Parse JSON in Python.", '```python\nimport json\ndata = json.loads(\'{"ok": true}\')\n```'),
    ("Create JSON in Python.", '```python\nimport json\ntext = json.dumps({"ok": True})\n```'),
    ("Make a Python class.", '```python\nclass Counter:\n    def __init__(self):\n        self.value = 0\n```'),
    ("Write a basic Python test for add.", '```python\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\n```'),
    ("Fix Python: print(name) gives NameError.", 'Define name before using it:\n```python\nname = "Sam"\nprint(name)\n```'),
    ("Fix Python: if x = 3:", 'Use == for comparison:\n```python\nif x == 3:\n    print("Three")\n```'),
    ("Fix Python: numbers[10] raises IndexError.", 'The index is outside the list. Check len(numbers) before accessing an item.'),
    ("Fix Python: int('hello') raises ValueError.", 'The text is not a valid integer. Validate it or catch ValueError.'),
    ("Remove whitespace in Python.", '```python\nclean = text.strip()\n```'),
    ("Convert text to lowercase in Python.", '```python\nlower = text.lower()\n```'),
    ("Join strings in Python.", '```python\ntext = ", ".join(["red", "blue"])\n```'),
    ("Split a string in Python.", '```python\nwords = text.split()\n```'),
    ("Write a JavaScript square function.", '```javascript\nconst square = x => x * x;\n```'),
    ("Loop over a JavaScript array.", '```javascript\nfor (const item of items) {\n  console.log(item);\n}\n```'),
    ("Double a JavaScript array.", '```javascript\nconst doubled = numbers.map(n => n * 2);\n```'),
    ("Filter even numbers in JavaScript.", '```javascript\nconst even = numbers.filter(n => n % 2 === 0);\n```'),
    ("Sum numbers in JavaScript.", '```javascript\nconst total = numbers.reduce((sum, n) => sum + n, 0);\n```'),
    ("Parse JSON in JavaScript.", '```javascript\nconst data = JSON.parse(\'{"ok":true}\');\n```'),
    ("Make a JavaScript object.", '```javascript\nconst user = { name: "Sam", age: 20 };\n```'),
    ("Write an HTML button.", '```html\n<button type="button">Click me</button>\n```'),
    ("Write an HTML heading.", '```html\n<h1>Hello, world!</h1>\n```'),
    ("Write an HTML input.", '```html\n<label>Name <input name="name" type="text"></label>\n```'),
    ("Center content with CSS.", '```css\n.container {\n  display: flex;\n  align-items: center;\n  justify-content: center;\n}\n```'),
    ("Make a CSS grid with two columns.", '```css\n.grid {\n  display: grid;\n  grid-template-columns: 1fr 1fr;\n}\n```'),
    ("Set a CSS background color.", '```css\nbody {\n  background: #101114;\n  color: white;\n}\n```'),
]

for name, operator in (("subtract", "-"), ("multiply", "*"), ("divide", "/")):
    pairs.append((f"Write a Python {name} function.", f"```python\ndef {name}(a, b):\n    return a {operator} b\n```"))
    pairs.append((f"Write a JavaScript {name} function.", f"```javascript\nconst {name} = (a, b) => a {operator} b;\n```"))

records = [{"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]} for question, answer in pairs]
records.append({"messages": [
    {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hi! What would you like to build?"},
    {"role": "user", "content": "a Python function"}, {"role": "assistant", "content": "What should the function do?"}
]})
records.append({"messages": [
    {"role": "user", "content": "Write a Python add function."}, {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"},
    {"role": "user", "content": "test it"}, {"role": "assistant", "content": "```python\nassert add(2, 3) == 5\n```"}
]})

if __name__ == "__main__":
    target = Path(__file__).with_name("starter.jsonl")
    target.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(f"Wrote {len(records)} original starter conversations to {target}")
