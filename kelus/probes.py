"""Independent held-out probe set for the wider benchmark.

Written by hand, separately from every data generator, using natural wording,
plurals, slang, typos, and sentence shapes the curricula do not template.
Training generators must never import this module; `kelus.benchmark` asserts
that no probe prompt appears verbatim in a training split.

Each intent lists every canonical reply the curricula use for it. A chat probe
passes only when the reply equals one of those replies exactly. A Python
function probe passes when the bounded AST interpreter in `kelus.evaluate`
confirms the function on several inputs, so different correct code is accepted.
"""

GREETING = "Hi! I'm Kelus Model B1. What would you like to build?"
IDENTITY = "I'm Kelus Model B1, a small model trained locally from scratch for basic chat and code."
CLARIFY = ("I may have misunderstood. What would you like me to explain?",
           "I misunderstood your request. Please restate what you need, and I will try to answer that specific question.")
HELP = ("Sure. Tell me what you want to build and which programming language you want to use.",
        "Yes. I can write small Python and JavaScript examples. Tell me what you want to build and which language to use.")
UNKNOWN_PROGRAM = "I have not learned to write that program yet. I can write small examples like fibonacci, factorial, or reversing a string."
OUT_OF_SCOPE = ("That is outside what I can reliably answer. Could you ask a small coding question or describe one specific part?",
                "I do not have enough training to answer that reliably. I can help with a short coding example or a basic programming concept.",
                UNKNOWN_PROGRAM)

INTENTS = {
    "greeting": (GREETING,),
    "identity": (IDENTITY,),
    "how_are_you": ("I'm ready to help. What are you working on?",),
    "thanks": ("You're welcome! What would you like to try next?",),
    "goodbye": ("See you! Happy coding.",),
    "clarify": CLARIFY,
    "help": HELP,
    "abilities": ("I can practice basic chat, explain coding concepts, and write small Python or JavaScript examples. I'm experimental and can make mistakes.",) + HELP[1:],
    "training": ("My weights were initialized randomly and trained on local examples. My transformer and gradients are written here; NumPy handles array math.",),
    "reliability": ("I can make mistakes. Review the code and test it before using it.",),
    "text_only": ("I generate text only. I cannot run code, browse the web, or edit your files.",),
    "debug_request": ("Share the code, the exact error message, and what you expected to happen. Then we can look for the problem.",),
    "debug_method": ("Reproduce the problem, read the error message, and test one small change at a time.",),
    "learning": ("Start with variables, conditions, loops, and functions. Practice by writing small programs and testing them.",),
    "joke": ("Why did the programmer bring a ladder? To reach a higher level!",),
    "minecraft": ("Minecraft is a sandbox game where players build, explore, and survive in a world of blocks.",),
    "minecraft_fps": ("For Minecraft performance, try lower render distance, fewer particles, and lighter graphics settings. Measure FPS before and after each change.",),
    "out_of_scope": OUT_OF_SCOPE,
    "concept:variable": ("A variable is a name that refers to a value.",),
    "concept:function": ("A function is a reusable block of code that takes inputs and can return a result.",),
    "concept:loop": ("A loop repeats a block of code.",),
    "concept:list": ("A list stores an ordered collection of values.",),
    "concept:dictionary": ("A dictionary maps keys to values.",),
    "concept:boolean": ("A boolean is a value that is either True or False.",),
    "concept:string": ("A string is text enclosed in quotes.",),
    "concept:integer": ("An integer is a whole number without a fractional part.",),
    "concept:condition": ("A condition is an expression that evaluates to True or False.",),
    "concept:return statement": ("A return statement sends a result back from a function.",),
    "concept:syntax error": ("A syntax error means the code does not follow the language's grammar.",),
    "concept:test": ("A test checks that code behaves as expected for a given input.",),
    "concept:comment": ("A comment explains code and is ignored when the program runs.",),
    "concept:recursion": ("Recursion is when a function calls itself. It needs a base case to stop.",),
    "concept:algorithm": ("An algorithm is a sequence of steps for solving a problem.",),
    "concept:exception": ("An exception is an error that interrupts normal program execution.",),
    "concept:Python": ("Python is a programming language used for scripts, apps, and data work.",),
    "concept:JavaScript": ("JavaScript is a programming language often used to add behavior to web pages.",),
    "concept:HTML": ("HTML describes the structure of a web page.",),
    "concept:CSS": ("CSS controls how a web page looks.",),
    "code:hello": ('```python\nprint("Hello, world!")\n```',),
    "code:sort": ("```python\nnumbers = [3, 1, 2]\nordered = sorted(numbers)\n```",),
    "code:loop": ("```python\nfor i in range(5):\n    print(i)\n```",),
    "js:add": ("```javascript\nfunction add(a, b) {\n  return a + b;\n}\n```",),
    "js:square": ("```javascript\nfunction square(x) {\n  return x * x;\n}\n```",),
    "js:hello": ('```javascript\nconsole.log("Hello, world!");\n```',),
    "js:double": ("```javascript\nfunction double(x) {\n  return x * 2;\n}\n```",),
    "code:fibonacci": ("```python\ndef fibonacci(n):\n    numbers = [0, 1]\n    while len(numbers) < n:\n        numbers.append(numbers[-1] + numbers[-2])\n    return numbers[:n]\n```",),
    "code:factorial": ("```python\ndef factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```",),
    "code:is_prime": ("```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```",),
    "code:fizzbuzz": ('```python\nfor i in range(1, 16):\n    if i % 15 == 0:\n        print("FizzBuzz")\n    elif i % 3 == 0:\n        print("Fizz")\n    elif i % 5 == 0:\n        print("Buzz")\n    else:\n        print(i)\n```',),
    "unknown_program": (UNKNOWN_PROGRAM,) + OUT_OF_SCOPE[:2],
}

CHAT = {
    "greeting": ["howdy", "greetings!", "hey hey", "yo kelus whats good", "morning!", "hello hello", "heyo", "good day to you", "hellooo kelus", "hi again"],
    "identity": ["who am i talking to?", "what are you exactly", "are you a bot?", "do you have a name?", "who's this", "what should i know about you", "whats ur name", "which model is this?"],
    "how_are_you": ["how you doing?", "how's your day going", "how are u", "how have you been?", "you doing ok?", "hows everything"],
    "thanks": ["cheers!", "much appreciated", "perfect, thanks", "thank you kindly", "thanks a bunch", "ty so much", "great, thanks!", "awesome thank u"],
    "goodbye": ["ok bye", "cya", "gotta go now", "goodbye for now!", "peace out", "catch you later", "i'm heading out, bye", "see you tomorrow"],
    "clarify": ["wut", "i'm confused", "say again?", "come again?", "what are you talking about", "eh?", "sorry, what?", "that doesn't make sense", "i dont follow"],
    "help": ["i need help with a program", "can you assist me?", "help me write a program", "assist me with some code please", "i want to make a program", "could you give me a hand with code?"],
    "abilities": ["what are you good at?", "what can i ask you?", "what are your skills", "list your abilities", "what do you do?", "what are you capable of?"],
    "training": ["were you pretrained?", "do you run on tensorflow?", "are you connected to openai?", "how did you learn?", "what library powers you?", "were you made with pytorch?"],
    "reliability": ["should i trust this code?", "do you ever make mistakes?", "can i rely on your answers?", "is your code always right?", "are your answers accurate?"],
    "text_only": ["can you run my program?", "please execute this script", "can you look at my files?", "can you google something for me?", "can you access the internet?", "open my project folder"],
    "debug_request": ["my script throws an error", "i got an error", "my code keeps failing", "something is broken in my code", "my program won't start", "my app keeps crashing, help"],
    "debug_method": ["what's the best way to debug?", "how do i track down a bug?", "any tips for debugging?", "how do you debug a program?", "how should i go about debugging"],
    "learning": ["how can i learn to code?", "i want to start programming", "teach me to program", "i'm new to coding", "where do i begin with python?", "how do beginners learn coding"],
    "joke": ["know any jokes?", "tell me something funny", "joke please", "got a programming joke?", "make me smile with a joke"],
    "minecraft": ["what's minecraft about?", "explain minecraft", "have you heard of minecraft?", "what do you do in minecraft"],
    "minecraft_fps": ["minecraft is so laggy", "how do i get better fps in minecraft", "my minecraft runs slow", "boost minecraft fps", "minecraft keeps stuttering"],
    "out_of_scope": ["who is the best football player?", "what's the weather like?", "write me a whole novel", "what is the meaning of life", "what is the bitcoin price", "who will win the world cup"],
}

CONCEPT_PROBES = {
    "variable": ["what are variables?", "variables in programming?", "explain variables to me"],
    "function": ["what are functions?", "i don't understand functions", "how do functions work"],
    "loop": ["what are loops?", "how do loops work", "explain loops please"],
    "list": ["what are lists in python?", "how do lists work"],
    "dictionary": ["what are dictionaries?", "what's a dict in python"],
    "boolean": ["what are booleans?", "what does bool mean"],
    "string": ["what are strings?", "what's a string in coding"],
    "integer": ["what are integers?", "what's an int"],
    "condition": ["what are conditions in code?", "how do conditions work"],
    "return statement": ["what does return do?", "what are return statements"],
    "syntax error": ["what are syntax errors?", "why do i get a syntax error"],
    "test": ["what are tests in programming?", "what's a unit test"],
    "comment": ["what are comments in code?", "how do comments work"],
    "recursion": ["how does recursion work?", "recursion?"],
    "algorithm": ["what are algorithms?", "explain algorithms"],
    "exception": ["what are exceptions?", "how do exceptions work"],
    "Python": ["what's python?", "tell me about the python language"],
    "JavaScript": ["what's javascript?", "what is js"],
    "HTML": ["what's html?", "what is html used for"],
    "CSS": ["what's css?", "what is css used for"],
}

# Python function probes: (prompt, function name, trials).
CODE_PROBES = [
    ("python function to add 2 numbers", "add", [((2, 3), 5), ((-4, 4), 0)]),
    ("how do i add two numbers in python?", "add", [((2, 3), 5), ((0, 9), 9)]),
    ("code that subtracts two numbers in python", "subtract", [((5, 3), 2), ((0, 4), -4)]),
    ("python: multiply two numbers", "multiply", [((3, 4), 12), ((-2, 5), -10)]),
    ("write a function that squares numbers", "square", [((3,), 9), ((-4,), 16)]),
    ("python code to square something", "square", [((3,), 9), ((0,), 0)]),
    ("make a function that doubles a number", "double", [((3,), 6), ((-1,), -2)]),
    ("python check if n is even", "is_even", [((2,), True), ((3,), False)]),
    ("how to tell if a number is odd in python", "is_odd", [((3,), True), ((4,), False)]),
    ("how do i reverse a string in python?", "reverse", [(("abc",), "cba"), (("",), "")]),
    ("python function for the length of a list", "length", [(([1, 2],), 2), (([],), 0)]),
    ("add up all numbers in a list with python", "total", [(([1, 2, 3],), 6), (([],), 0)]),
    ("python function that greets a person by name", "greet", [(("Sam",), "Hello, Sam!")]),
]

# Fixed-answer code probes: (prompt, intent).
CODE_EXACT = [
    ("python program that prints hello world", "code:hello"),
    ("how do i print hello world in python?", "code:hello"),
    ("how to sort a list in python", "code:sort"),
    ("python for loop example please", "code:loop"),
    ("javascript function to add two numbers", "js:add"),
    ("js function that squares a number", "js:square"),
    ("hello world javascript", "js:hello"),
    ("javascript code to double a number", "js:double"),
    ("give me fibonacci numbers in python", "code:fibonacci"),
    ("how do i calculate a factorial in python?", "code:factorial"),
    ("python prime number checker", "code:is_prime"),
    ("can you do fizzbuzz in python", "code:fizzbuzz"),
    ("write a tetris clone in python", "unknown_program"),
    ("i want to build a recipe website", "unknown_program"),
]

# Python function probes added with the expanded snippet set.
CODE_PROBES_V3 = [
    ("python function that finds the biggest number in a list", "maximum", [(([3, 9, 2],), 9), (([-1],), -1)]),
    ("get the smallest value from a list in python", "minimum", [(([3, 9, 2],), 2)]),
    ("how do i check if a word is a palindrome in python", "is_palindrome", [(("level",), True), (("abc",), False)]),
    ("convert celsius into fahrenheit with python", "celsius_to_fahrenheit", [((100,), 212.0), ((0,), 32.0)]),
]

SQUARE = "```python\ndef square(x):\n    return x * x\n```"
VARIABLE = "A variable is a name that refers to a value."
LOOP = "A loop repeats a block of code."

# Two-turn probes with one preceding exchange, matching the deployed runtime.
MULTI = [
    ([("python square function please", SQUARE)], "yo, hi again", "greeting"),
    ([("python square function please", SQUARE)], "thanks a bunch", "thanks"),
    ([("python square function please", SQUARE)], "wait im lost", "clarify"),
    ([("what are variables?", VARIABLE)], "can i see that in code?", "example:variable"),
    ([("explain loops please", LOOP)], "show me some code for it", "example:loop"),
    ([("what are variables?", VARIABLE)], "cool thanks", "thanks"),
    ([("hey", GREETING)], "what are loops?", "concept:loop"),
    ([("hey", GREETING)], "write a python function to square a number", "code:square"),
    # The reported app sequence: a vague request, then a specific one.
    ([("I want to build a scale", HELP[0])], "i want to build a fibonacci scale using python!", "code:fibonacci"),
    ([("hey", GREETING)], "factorial in python please", "code:factorial"),
]

EXAMPLES = {
    "example:variable": ("```python\nx = 5\n```",),
    "example:loop": ("```python\nfor i in range(3):\n    print(i)\n```",),
}

# Context regression (reported separately, not held out): simple trained
# messages after each kind of earlier exchange. The earlier reply must not
# capture the answer, e.g. "hi!" after "What would you like me to explain?".
CONTEXT_PRIORS = [("hello", GREETING), ("who are you", IDENTITY), ("how are you", INTENTS["how_are_you"][0]),
                  ("thanks", INTENTS["thanks"][0]), ("bye", INTENTS["goodbye"][0]), ("what", CLARIFY[0]), ("no", CLARIFY[1]),
                  ("help", HELP[0]), ("can you code", HELP[1]), ("what can you do", INTENTS["abilities"][0]),
                  ("how were you trained", INTENTS["training"][0]), ("can i trust your code", INTENTS["reliability"][0]),
                  ("can you run code", INTENTS["text_only"][0]), ("my code is broken", INTENTS["debug_request"][0]),
                  ("how do i debug", INTENTS["debug_method"][0]), ("teach me coding", INTENTS["learning"][0]),
                  ("tell me a joke", INTENTS["joke"][0]), ("what is minecraft", INTENTS["minecraft"][0]),
                  ("improve minecraft fps", INTENTS["minecraft_fps"][0]), ("what is the weather", OUT_OF_SCOPE[1]),
                  ("make me a snake game", UNKNOWN_PROGRAM), ("what is a loop", LOOP), ("Write a Python square function.", SQUARE)]
CONTEXT_FOLLOW_UPS = [("hi!", "greeting"), ("hello", "greeting"), ("hello!", "greeting"), ("hey", "greeting"),
                      ("thanks", "thanks"), ("bye", "goodbye"), ("what", "clarify"), ("who are you", "identity")]
