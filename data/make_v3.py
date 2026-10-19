"""Curriculum v3: broader wording for the same practiced intents and replies.

Still original synthetic practice data on a closed set of topics, not broad
pretraining. Compared with v1/v2 it adds:

- many more hand-written phrasings per intent, plus plural/alias concept forms;
- compositional templates for concepts and code (infinitive, "function that",
  noun, and gerund requests, with or without a language name);
- light noise: casing, punctuation, dropped apostrophes, txt-speak, one typo;
- histories of one earlier exchange, matching the deployed runtime, with
  extra short chat turns after every kind of earlier exchange;
- corrections after a wrong earlier reply, which is context only
  ("supervise": false) and never a training target;
- more Python snippets (fibonacci, factorial, primes, min/max, average,
  palindromes, vowels, temperature, fizzbuzz), "build a ..." requests, and an
  honest decline for specific programs B1 has not learned.

Decontamination: every probe in kelus/probes.py and every held-out prompt in
the older evaluators is removed. So is any prompt that contains a multi-word
probe, or asks for the same two or more content words as a probe once filler
and question words are ignored (allowing one extra or missing word when at
least three remain). The probes are read only for this removal.
"""
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kelus import probes, evaluate, evaluate_repair  # noqa: E402

GREETING = "Hi! I'm Kelus Model B1. What would you like to build?"
IDENTITY = "I'm Kelus Model B1, a small model trained locally from scratch for basic chat and code."
CONFUSION = "I may have misunderstood. What would you like me to explain?"
CORRECTION = "I misunderstood your request. Please restate what you need, and I will try to answer that specific question."
HELP = "Sure. Tell me what you want to build and which programming language you want to use."
CAN_CODE = "Yes. I can write small Python and JavaScript examples. Tell me what you want to build and which language to use."
OUT_OF_SCOPE = "I do not have enough training to answer that reliably. I can help with a short coding example or a basic programming concept."
UNKNOWN_PROGRAM = "I have not learned to write that program yet. I can write small examples like fibonacci, factorial, or reversing a string."

CHAT = {
    GREETING: ["hello", "hi", "hey", "hiya", "heya", "hii", "hiii", "helo", "hai", "hi there", "hello there", "hey there",
               "hi kelus", "hello kelus", "hey kelus", "good morning", "good afternoon", "good evening", "sup", "what's up",
               "wassup", "yo", "yo yo", "hey buddy", "hi friend", "hello friend", "hey bro", "hi bot", "hello bot", "hey man",
               "hey y'all", "nice to meet you", "hi, nice to meet you", "hello again", "hey again", "good to see you", "morning kelus"],
    IDENTITY: ["who are you", "what are you", "what is your name", "what's your name", "tell me your name", "introduce yourself",
               "what model are you", "are you kelus", "who made you", "who created you", "who built this model",
               "what's this model called", "your name please", "what do i call you", "are you an ai", "are you a chatbot",
               "what kind of ai are you", "tell me about yourself", "who exactly are you", "what model am i using", "is this kelus",
               "which ai is this", "what are you called", "say who you are", "what is b1", "are you b1"],
    "I'm ready to help. What are you working on?":
              ["how are you", "how's it going", "how are you doing", "you good", "are you ready", "how are things", "how is it going",
               "how do you feel", "are you okay", "how's life", "what's new with you", "how are you feeling today", "you alright",
               "all good with you", "are you doing well", "how is your day"],
    "You're welcome! What would you like to try next?":
              ["thanks", "thank you", "thank you so much", "thanks a lot", "thanks so much", "ty", "thx", "tysm", "thanks kelus",
               "thank you kelus", "that helped", "that helps", "nice thanks", "cool thanks", "great thank you", "appreciate it",
               "i appreciate it", "i appreciate your help", "thanks for the help", "thanks for helping", "thanks man", "thanks bro",
               "awesome", "nice", "cool", "perfect", "great job", "good job", "that works", "that's helpful", "very helpful",
               "you're the best", "nice one", "sweet"],
    "See you! Happy coding.":
              ["bye", "goodbye", "see you", "see ya", "later", "good night", "bye bye", "see you later", "talk later", "talk to you soon",
               "i'm going now", "i have to go", "gotta run", "farewell", "night", "goodnight", "bye kelus", "later kelus", "see you soon",
               "i'm done for today", "that's all for now", "signing off", "ttyl", "i'm off"],
    CONFUSION: ["what", "what?", "huh", "???", "?", "hmm?", "i don't understand", "that makes no sense", "what do you mean", "explain that",
                "you are speaking gibberish", "that's gibberish", "can you explain", "what was that", "i don't get it", "what does that mean",
                "i'm lost", "that's confusing", "you lost me", "could you clarify", "please clarify", "what do you mean by that",
                "what is that supposed to mean", "pardon", "what did you say", "can you rephrase", "i do not understand that"],
    CORRECTION: ["no", "nope", "nah", "wrong", "that's wrong", "that is wrong", "you are wrong", "not what i asked", "you misunderstood",
                 "that's not it", "try again", "i meant something else", "that was unrelated", "you did not answer", "answer my question",
                 "stop repeating yourself", "that's not right", "incorrect", "no that's not it", "not quite", "that's not what i wanted",
                 "bruh", "bro what", "nooo", "no no no", "that's the wrong answer", "you got it wrong", "that isn't what i asked for"],
    HELP: ["help", "help me", "can you help", "can you help me", "i need help", "help me code", "let's build something", "i want to make an app",
           "i want to build something", "let's code something", "help me build an app", "i need help coding", "can you help with my code",
           "help me with a project", "help me make a website", "let's write some code", "i have a coding project",
           "could you help with programming", "i'd like help with code", "i want to create a game"],
    CAN_CODE: ["can you code", "do you code", "can you program", "can you write code", "can you write programs", "do you know how to code",
               "are you able to code", "do you know python", "do you know javascript", "can you write python", "can you write javascript",
               "can u code", "are you good at coding", "do you do programming", "can you make programs", "can you code in python",
               "can you code in javascript", "you know how to program"],
    "I can practice basic chat, explain coding concepts, and write small Python or JavaScript examples. I'm experimental and can make mistakes.":
              ["what can you do", "what are your abilities", "can you chat", "what do you know", "are you smart", "are you a coding assistant",
               "what are your features", "what can you help with", "what can you help me with", "what do you know how to do",
               "show me what you can do", "what are your limits", "what tasks can you handle", "what is your purpose", "how can you help me",
               "what are you for", "what kind of questions can you answer"],
    "My weights were initialized randomly and trained on local examples. My transformer and gradients are written here; NumPy handles array math.":
              ["are you trained from scratch", "are you from scratch", "how were you trained", "do you use a framework", "what framework do you use",
               "do you use pytorch", "do you use an api", "are you using chatgpt", "are you a pretrained model", "are you chatgpt", "how were you made",
               "how do you work", "what are you built with", "are you built on gpt", "did you use pretrained weights", "what is your architecture",
               "how was this model trained", "do you use keras", "are you based on llama", "do you call an external api",
               "is this a wrapper around chatgpt", "are you running locally"],
    "I can make mistakes. Review the code and test it before using it.":
              ["can i trust your code", "is your code correct", "are you always right", "will this code work", "can you make mistakes",
               "is this reliable", "are you reliable", "is this answer right", "can you be wrong", "do you get things wrong",
               "should i double check your code", "is your output safe to use", "how accurate are you", "do your answers have errors"],
    "I generate text only. I cannot run code, browse the web, or edit your files.":
              ["can you run code", "run this code", "execute my code", "can you open files", "can you browse the internet", "search the web",
               "can you edit my project", "can you see my screen", "can you install packages", "run my tests", "can you execute python",
               "look up something online", "can you modify my code directly", "can you open a website", "deploy my app",
               "can you use the terminal", "can you save files"],
    "Share the code, the exact error message, and what you expected to happen. Then we can look for the problem.":
              ["my code is broken", "my code does not work", "my code doesn't work", "it crashes", "there is an error", "help me debug",
               "fix my code", "it isn't working", "why is my app crashing", "i have a bug", "my code has a bug", "i'm getting an exception",
               "my program crashed", "this code fails", "my tests are failing", "can you fix this bug", "my function doesn't work",
               "the program gives the wrong output", "why doesn't my code work"],
    "Reproduce the problem, read the error message, and test one small change at a time.":
              ["how do i debug", "how to debug code", "debugging tips", "how do i find a bug", "how should i fix a bug", "how do i fix bugs",
               "what is a good debugging process", "how can i debug faster", "steps to debug a program", "how do programmers find bugs",
               "how to fix errors in code", "best debugging strategy", "how to troubleshoot code"],
    "Start with variables, conditions, loops, and functions. Practice by writing small programs and testing them.":
              ["i want to learn coding", "teach me coding", "how do i start coding", "i am new to programming", "i'm a beginner",
               "how to learn programming", "teach me python", "how do i get into programming", "i want to become a programmer",
               "where should i start learning to code", "best way to learn python", "i've never coded before",
               "how can i get better at coding", "what should i learn first in programming", "teach me javascript", "help me learn python",
               "i'm just starting to code"],
    "Why did the programmer bring a ladder? To reach a higher level!":
              ["tell me a joke", "say something funny", "coding joke", "make me laugh", "tell a programming joke", "tell me a coding joke",
               "i need a laugh", "programmer joke", "joke time", "tell me a dev joke", "i'm bored, tell me a joke", "a joke about code"],
    "Minecraft is a sandbox game where players build, explore, and survive in a world of blocks.":
              ["what is minecraft", "do you know minecraft", "tell me about minecraft", "what's minecraft", "describe minecraft",
               "is minecraft fun", "how do you play minecraft", "what is the game minecraft", "what kind of game is minecraft"],
    "For Minecraft performance, try lower render distance, fewer particles, and lighter graphics settings. Measure FPS before and after each change.":
              ["what is lunar client", "what is badlion client", "how do i get more minecraft fps", "improve minecraft fps", "minecraft is lagging",
               "how to optimize minecraft", "minecraft lag fix", "make minecraft run faster", "minecraft low fps", "increase fps in minecraft",
               "minecraft performance tips", "my minecraft frame rate is bad", "minecraft freezes a lot", "best minecraft settings for fps"],
    OUT_OF_SCOPE: ["what's the capital of japan", "who won the game last night", "write a 500 page book", "how do i fix my car",
                   "what should i eat for dinner", "give me stock tips", "predict the lottery numbers", "what's the population of earth",
                   "translate this into french", "who is the president", "explain black holes", "how tall is mount everest",
                   "what's the temperature outside", "diagnose my illness", "give me legal advice", "who will win the election",
                   "what happened in the news", "how far is the moon", "summarize world history",
                   "explain quantum physics", "train a large ai", "do my homework", "tell me everything",
                   "asdfgh", "qwerty", "zxqv", "lkjhg", "sdfsdf", "what is the weather", "what is the latest news", "explain general relativity"],
}

# topic, explanation, follow-up example, singular forms, plural forms, mass-noun forms
CONCEPTS = [
    ("variable", "A variable is a name that refers to a value.", "```python\nx = 5\n```", ["variable", "var"], ["variables", "vars"], []),
    ("function", "A function is a reusable block of code that takes inputs and can return a result.", "```python\ndef double(x):\n    return x * 2\n```", ["function", "func"], ["functions"], []),
    ("loop", "A loop repeats a block of code.", "```python\nfor i in range(3):\n    print(i)\n```", ["loop", "while loop"], ["loops", "while loops"], ["looping", "iteration"]),
    ("list", "A list stores an ordered collection of values.", "```python\nitems = [1, 2, 3]\n```", ["list", "python list", "array"], ["lists", "arrays"], []),
    ("dictionary", "A dictionary maps keys to values.", '```python\nperson = {"name": "Sam"}\n```', ["dictionary", "dict", "hash map"], ["dictionaries", "dicts", "hash maps"], []),
    ("boolean", "A boolean is a value that is either True or False.", "```python\nready = True\n```", ["boolean", "bool"], ["booleans", "bools"], []),
    ("string", "A string is text enclosed in quotes.", '```python\nname = "Sam"\n```', ["string", "str"], ["strings"], []),
    ("integer", "An integer is a whole number without a fractional part.", "```python\ncount = 10\n```", ["integer", "int", "whole number"], ["integers", "ints", "whole numbers"], []),
    ("condition", "A condition is an expression that evaluates to True or False.", '```python\nif score > 10:\n    print("You win!")\n```', ["condition", "conditional", "if statement"], ["conditions", "conditionals", "if statements"], []),
    ("return statement", "A return statement sends a result back from a function.", "```python\ndef add(a, b):\n    return a + b\n```", ["return statement", "return value"], ["return statements", "return values"], ["return", "the return keyword"]),
    ("syntax error", "A syntax error means the code does not follow the language's grammar.", '```python\nprint("Hello")\n```', ["syntax error"], ["syntax errors"], []),
    ("test", "A test checks that code behaves as expected for a given input.", "```python\nassert 2 + 3 == 5\n```", ["test", "unit test"], ["tests", "unit tests"], ["testing", "software testing"]),
    ("comment", "A comment explains code and is ignored when the program runs.", "```python\n# This is a comment\n```", ["comment", "code comment"], ["comments", "code comments"], []),
    ("recursion", "Recursion is when a function calls itself. It needs a base case to stop.", "```python\ndef countdown(n):\n    if n > 0:\n        countdown(n - 1)\n```", ["recursive function"], ["recursive functions"], ["recursion"]),
    ("algorithm", "An algorithm is a sequence of steps for solving a problem.", "```python\nnumbers = [3, 1, 2]\nordered = sorted(numbers)\n```", ["algorithm"], ["algorithms"], []),
    ("exception", "An exception is an error that interrupts normal program execution.", '```python\nraise ValueError("Invalid value")\n```', ["exception", "runtime error"], ["exceptions", "runtime errors"], []),
    ("Python", "Python is a programming language used for scripts, apps, and data work.", '```python\nprint("Hello, world!")\n```', [], [], ["python", "Python", "the python language", "the python programming language"]),
    ("JavaScript", "JavaScript is a programming language often used to add behavior to web pages.", '```javascript\nconsole.log("Hello, world!");\n```', [], [], ["javascript", "JavaScript", "JS", "js"]),
    ("HTML", "HTML describes the structure of a web page.", "```html\n<h1>Hello</h1>\n```", [], [], ["html", "HTML"]),
    ("CSS", "CSS controls how a web page looks.", "```css\nbody { color: green; }\n```", [], [], ["css", "CSS"]),
]

SINGULAR = ["what is {a}", "what's {a}", "whats {a}", "explain {a}", "define {a}", "what does {a} do", "describe {a}",
            "what is {a} in programming", "what is {a} in python", "how does {a} work", "tell me what {a} is",
            "i want to know what {a} is", "meaning of {a}", "explain what {a} is", "what is {a} used for", "why would i use {a}",
            "{t}?", "{t} meaning", "define {t}", "explain {t}", "what is {t}", "{t} explained", "eli5 {t}", "{t} in programming",
            "teach me about {t}", "tell me about {t}"]
PLURAL = ["explain {ts}", "what are {ts}", "what r {ts}", "how do {ts} work", "{ts}?", "tell me about {ts}", "i don't understand {ts}",
          "help me understand {ts}", "what are {ts} used for", "what are {ts} in programming", "what are {ts} in python",
          "teach me about {ts}", "{ts} explained", "can you explain {ts}", "what do {ts} do", "why do we use {ts}", "i'm confused about {ts}"]
MASS = ["what is {m}", "what's {m}", "explain {m}", "{m}?", "tell me about {m}", "define {m}", "how does {m} work", "describe {m}",
        "what is {m} used for", "i want to learn what {m} is", "can you explain {m}", "{m} explained", "what does {m} mean"]

FOLLOW_UPS = ["show me an example", "give me an example", "example please", "example?", "can you show an example",
              "how does that look in code", "what does that look like in code", "show me the code", "code example please",
              "write an example", "i want to see an example", "show me how it looks", "and in code?", "show me one",
              "got an example?", "what would that look like", "example in code please", "demonstrate it"]

# name, code, infinitive, third person ("a function that ..."), noun, gerund
SNIPPETS = [
    ("add", "def add(a, b):\n    return a + b",
     ["add two numbers", "add 2 numbers", "add a and b", "add numbers together", "sum two numbers", "find the sum of two numbers"],
     ["adds two numbers", "adds 2 numbers", "returns the sum of two numbers", "adds a and b"],
     ["addition function", "add function", "function for adding two numbers", "adder function"], ["adding two numbers", "adding 2 numbers together"]),
    ("subtract", "def subtract(a, b):\n    return a - b",
     ["subtract two numbers", "subtract b from a", "find the difference of two numbers", "subtract one number from another"],
     ["subtracts two numbers", "returns the difference of two numbers", "subtracts b from a"],
     ["subtraction function", "subtract function", "difference function"], ["subtracting two numbers", "subtracting numbers"]),
    ("multiply", "def multiply(a, b):\n    return a * b",
     ["multiply two numbers", "multiply 2 numbers", "multiply a by b", "find the product of two numbers"],
     ["multiplies two numbers", "returns the product of two numbers", "multiplies a and b"],
     ["multiplication function", "multiply function", "product function"], ["multiplying two numbers"]),
    ("square", "def square(x):\n    return x * x",
     ["square a number", "square x", "square an integer", "get the square of a number", "compute x squared", "raise a number to the power of 2"],
     ["squares a number", "returns the square of a number", "squares x", "returns x squared"],
     ["square function", "squaring function", "function for squaring"], ["squaring a number", "squaring numbers"]),
    ("double", "def double(x):\n    return x * 2",
     ["double a number", "double x", "multiply a number by 2", "multiply a number by two", "make a number twice as big"],
     ["doubles a number", "doubles x", "returns twice a number", "multiplies a number by 2"],
     ["double function", "doubling function"], ["doubling a number"]),
    ("is_even", "def is_even(n):\n    return n % 2 == 0",
     ["check if a number is even", "check whether a number is even", "test if n is even", "tell whether a number is even", "see if a number is even"],
     ["checks if a number is even", "returns True for even numbers", "tests whether n is even"],
     ["is_even function", "even number checker", "even check function", "even number function"], ["checking if a number is even"]),
    ("is_odd", "def is_odd(n):\n    return n % 2 != 0",
     ["check if a number is odd", "check whether a number is odd", "test if n is odd", "see if a number is odd"],
     ["checks if a number is odd", "returns True for odd numbers", "tests whether n is odd"],
     ["is_odd function", "odd number checker", "odd check function", "odd number function"], ["checking if a number is odd"]),
    ("reverse", "def reverse(text):\n    return text[::-1]",
     ["reverse a string", "reverse text", "reverse some text", "flip a string backwards", "reverse the characters in a string"],
     ["reverses a string", "returns a string backwards", "reverses text"],
     ["string reverse function", "reverse function", "string reverser"], ["reversing a string", "reversing text"]),
    ("length", "def length(items):\n    return len(items)",
     ["get the length of a list", "count the items in a list", "find how many items a list has", "get list length"],
     ["returns the length of a list", "counts items in a list"],
     ["list length function", "length function"], ["getting the length of a list", "counting list items"]),
    ("total", "def total(values):\n    return sum(values)",
     ["sum a list", "sum a list of numbers", "add up a list", "total a list of numbers", "get the sum of a list"],
     ["sums a list", "returns the total of a list", "adds up a list of numbers"],
     ["list sum function", "total function", "sum of list function"], ["summing a list", "adding up a list"]),
    ("greet", 'def greet(name):\n    return f"Hello, {name}!"',
     ["greet someone", "greet a person", "say hello to someone by name", "greet a user by name"],
     ["greets someone", "says hello to a name", "greets a user by name"],
     ["greeting function", "greet function"], ["greeting someone", "greeting a user"]),
    ("hello", 'print("Hello, world!")',
     ["print hello world", "say hello world", "print hello, world", "output hello world"], ["prints hello world"],
     ["hello world program", "hello world", "hello world example", "hello world script"], ["printing hello world"]),
    ("sort", "numbers = [3, 1, 2]\nordered = sorted(numbers)",
     ["sort a list", "sort numbers", "sort a list of numbers", "order a list", "put numbers in order"], ["sorts a list", "sorts numbers"],
     ["sorting example", "sort example", "list sorting snippet"], ["sorting a list", "sorting numbers"]),
    ("loop", "for i in range(5):\n    print(i)",
     ["print numbers from 0 to 4", "loop five times", "count from 0 to 4", "loop over a range"], ["loops five times", "prints 0 to 4"],
     ["for loop", "loop example", "for loop example", "range loop"], ["looping five times", "looping with range"]),
    ("fibonacci", "def fibonacci(n):\n    numbers = [0, 1]\n    while len(numbers) < n:\n        numbers.append(numbers[-1] + numbers[-2])\n    return numbers[:n]",
     ["print the fibonacci sequence", "generate fibonacci numbers", "get the first n fibonacci numbers", "make a fibonacci sequence", "calculate fibonacci numbers", "list fibonacci numbers"],
     ["generates fibonacci numbers", "returns the fibonacci sequence", "makes a fibonacci sequence", "returns the first n fibonacci numbers"],
     ["fibonacci function", "fibonacci sequence", "fibonacci generator", "fibonacci program", "fibonacci series", "fibonacci calculator", "fibonacci code"],
     ["generating fibonacci numbers", "computing the fibonacci sequence"]),
    ("factorial", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
     ["calculate a factorial", "compute the factorial of n", "find the factorial of a number", "get n factorial"],
     ["calculates a factorial", "returns the factorial of n", "computes n factorial"],
     ["factorial function", "factorial program", "factorial calculator"], ["calculating a factorial", "computing factorials"]),
    ("is_prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
     ["check if a number is prime", "test whether n is prime", "find out if a number is prime", "check for prime numbers"],
     ["checks if a number is prime", "returns True for prime numbers", "tests whether n is prime"],
     ["is_prime function", "prime checker", "prime number function", "primality test"], ["checking if a number is prime", "testing for primes"]),
    ("maximum", "def maximum(values):\n    return max(values)",
     ["find the largest number in a list", "get the maximum of a list", "find the max value in a list"],
     ["returns the largest number in a list", "finds the maximum value"], ["max function", "maximum function", "largest number finder"],
     ["finding the largest number in a list"]),
    ("minimum", "def minimum(values):\n    return min(values)",
     ["find the smallest number in a list", "get the minimum of a list", "find the min value in a list"],
     ["returns the smallest number in a list", "finds the minimum value"], ["min function", "minimum function", "smallest number finder"],
     ["finding the smallest number in a list"]),
    ("average", "def average(values):\n    return sum(values) / len(values)",
     ["find the average of a list", "calculate the mean of numbers", "average a list of numbers"],
     ["returns the average of a list", "calculates the mean"], ["average function", "mean function"], ["averaging a list"]),
    ("count_vowels", 'def count_vowels(text):\n    return sum(1 for char in text.lower() if char in "aeiou")',
     ["count the vowels in a string", "count vowels in text"], ["counts vowels in a string", "returns the number of vowels"],
     ["vowel counter", "count_vowels function"], ["counting vowels"]),
    ("is_palindrome", "def is_palindrome(text):\n    return text == text[::-1]",
     ["check if a word is a palindrome", "check whether a string is a palindrome", "test for a palindrome"],
     ["checks if text is a palindrome", "returns True for palindromes"], ["palindrome checker", "palindrome function"], ["checking for palindromes"]),
    ("celsius_to_fahrenheit", "def celsius_to_fahrenheit(celsius):\n    return celsius * 9 / 5 + 32",
     ["convert celsius to fahrenheit", "change celsius to fahrenheit", "turn celsius into fahrenheit"],
     ["converts celsius to fahrenheit"], ["temperature converter", "celsius to fahrenheit function"], ["converting celsius to fahrenheit"]),
    ("fizzbuzz", 'for i in range(1, 16):\n    if i % 15 == 0:\n        print("FizzBuzz")\n    elif i % 3 == 0:\n        print("Fizz")\n    elif i % 5 == 0:\n        print("Buzz")\n    else:\n        print(i)',
     ["solve fizzbuzz", "print fizzbuzz", "do fizzbuzz"], ["prints fizzbuzz"], ["fizzbuzz", "fizzbuzz program", "fizzbuzz solution"], ["solving fizzbuzz"]),
]
JS_SNIPPETS = [
    ("add", "function add(a, b) {\n  return a + b;\n}", ["add two numbers", "add 2 numbers", "sum two numbers"], ["adds two numbers", "returns the sum of two numbers"], ["add function", "addition function"], ["adding two numbers"]),
    ("square", "function square(x) {\n  return x * x;\n}", ["square a number", "square x", "get the square of a number"], ["squares a number", "returns the square of a number"], ["square function", "squaring function"], ["squaring a number"]),
    ("hello", 'console.log("Hello, world!");', ["print hello world", "log hello world", "say hello world"], ["prints hello world", "logs hello world"], ["hello world", "hello world program", "hello world example"], ["printing hello world"]),
    ("double", "function double(x) {\n  return x * 2;\n}", ["double a number", "double x", "multiply a number by 2"], ["doubles a number", "returns twice a number"], ["double function", "doubling function"], ["doubling a number"]),
]
INFINITIVE = ["write {L} code to {v}", "how do i {v} in {L}", "how to {v} in {L}", "{L}: {v}", "{v} in {L}", "{v} using {L}",
              "help me {v} with {L}", "show me how to {v} in {L}", "i want to {v} in {L}", "can you {v} in {L}",
              "{L} code to {v}", "write a {L} function to {v}", "make a {L} function to {v}", "need {L} code to {v}", "{L} snippet to {v}",
              "write {L} to {v}", "please write {L} code to {v}"]
THIRD = ["write a {L} function that {v}", "{L} function that {v}", "a {L} function that {v}", "give me a {L} function that {v}",
         "i need a {L} function that {v}", "create a {L} function which {v}", "can you write a {L} function that {v}",
         "write a function in {L} that {v}", "show me a {L} function that {v}"]
NOUN = ["write a {L} {v}", "{L} {v}", "{v} in {L}", "give me a {L} {v}", "make a {L} {v}", "show me a {L} {v}", "i need a {L} {v}",
        "{v} written in {L}", "can you make a {L} {v}", "{L} {v} please", "write me a {L} {v}",
        "i want to build a {v} using {L}", "i want to make a {v} with {L}", "build a {v} in {L}", "let's make a {v} in {L}"]
GERUND = ["{L} code for {v}", "example of {v} in {L}", "{v} in {L}", "code for {v} in {L}", "show me {v} in {L}", "{L} example of {v}"]
# Without a language name, code requests default to Python.
NO_LANGUAGE = [("infinitive", ["write code to {v}", "make a function to {v}", "code to {v}", "how do i write code to {v}", "write a function to {v}"]),
               ("third", ["write a function that {v}", "function that {v}", "i need a function that {v}", "create a function that {v}"]),
               ("noun", ["write a {v}", "give me a {v}", "{v} code", "show me a {v}", "i want to build a {v}"])]
# Specific programs B1 has not learned get an honest decline, not "tell me
# what you want to build" (the user already did). Generic requests such as
# "i want to build something" still ask for details.
UNKNOWN_TARGETS = ["a snake game", "a chess game", "a calculator app", "a todo app", "a discord bot", "a web scraper", "a music player",
                   "a chat app", "a login system", "a password manager", "a weather app", "a tic tac toe game", "an image editor",
                   "a budget tracker", "a quiz game", "a stopwatch app", "an alarm clock", "a minecraft mod", "a blog", "a rest api",
                   "a neural network", "a spreadsheet", "a paint program", "a platformer game", "a flappy bird clone", "a youtube downloader",
                   "an online store", "a social media app", "a file manager", "a text editor", "a pong game", "a calendar app", "a pdf reader",
                   "a voice assistant", "a stock trading bot", "a portfolio website", "a compiler", "a game engine", "an email client",
                   "a mobile app", "a 3d game", "a multiplayer server", "a crypto wallet", "a face recognition app", "a web browser",
                   "an operating system", "a complete minecraft client", "a web server with login and a database", "an entire app with payments"]
UNKNOWN_TEMPLATES = ["i want to build {x}", "i want to build {x} using {L}", "build {x} in {L}", "make me {x}", "write {x} in {L}", "code {x}",
                     "can you make {x}", "help me build {x} in {L}", "create {x} with {L}", "i want to make {x}", "write me {x}", "let's build {x}",
                     "can you code {x} in {L}", "i need {x}", "write the code for {x}", "make {x} in {L} for me"]
PYTHON = ["python", "Python", "python", "Python", "py"]
JAVASCRIPT = ["javascript", "JavaScript", "JS", "js"]

PREFIXES = [""] * 10 + ["hey ", "hey, ", "hi ", "ok ", "okay ", "so ", "um ", "uh ", "hmm ", "kelus ", "kelus, ", "yo ", "please ",
                        "quick question: ", "question: ", "hey kelus, ", "alright ", "btw ", "and "]
SUFFIXES = [""] * 8 + ["?", "?", "!", ".", " please", " pls", " plz", " thanks", " kelus", " :)", "??", "!!", "...", " lol", " now"]
SLANG = [(r"\byou\b", "u"), (r"\byour\b", "ur"), (r"\bare\b", "r"), (r"\bplease\b", "pls"), (r"\bthanks\b", "thx"), (r"\bokay\b", "ok"), (r"\bbecause\b", "cuz")]


def normalized(text):
    return re.sub(r"[^a-z0-9 ]", "", " ".join(text.casefold().split())).strip()


def article(word):
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


def typo(text, rng):
    words = text.split(" ")
    choices = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not choices:
        return text
    i = rng.choice(choices); w = words[i]; j = rng.randrange(1, len(w) - 1)
    kind = rng.randrange(3)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:] if kind == 0 else w[:j] + w[j + 1:] if kind == 1 else w[:j] + w[j] + w[j:]
    return " ".join(words)


def noisy(text, rng, wrap=True):
    if wrap:
        text = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
    if rng.random() < 0.3:
        text = text.replace("'", "").replace("’", "")
    if rng.random() < 0.15:
        for pattern, replacement in SLANG:
            text = re.sub(pattern, replacement, text)
    if rng.random() < 0.2:
        text = typo(text, rng)
    roll = rng.random()
    return text.upper() if roll < 0.05 else text[:1].upper() + text[1:] if roll < 0.4 else text


def variants(cores, rng, count):
    """Clean cores with simple casing/punctuation, then noisy wrapped samples."""
    found = []
    for core in cores:
        found += [core, core[:1].upper() + core[1:], core + "?" if not core.endswith("?") else core + "!"]
    attempts = 0
    while len(set(found)) < count + 3 * len(cores) and attempts < count * 20:
        found.append(noisy(rng.choice(cores), rng)); attempts += 1
    return list(dict.fromkeys(found))


def concept_cores(single, plural, mass):
    cores = [t.format(a=article(s), t=s) for s in single for t in SINGULAR]
    cores += [t.format(ts=p) for p in plural for t in PLURAL]
    cores += [t.format(m=m) for m in mass for t in MASS]
    return list(dict.fromkeys(cores))


def code_cores(languages, infinitive, third, noun, gerund, no_language):
    cores = []
    for forms, templates in ((infinitive, INFINITIVE), (third, THIRD), (noun, NOUN), (gerund, GERUND)):
        cores += [t.format(L=language, v=v) for v in forms for t in templates for language in languages]
    if no_language:
        lookup = {"infinitive": infinitive, "third": third, "noun": noun}
        cores += [t.format(v=v) for kind, templates in NO_LANGUAGE for v in lookup[kind] for t in templates]
    return list(dict.fromkeys(cores))


def forbidden():
    held = [p for prompts in probes.CHAT.values() for p in prompts]
    held += [p for prompts in probes.CONCEPT_PROBES.values() for p in prompts]
    held += [p for p, *_ in probes.CODE_PROBES + probes.CODE_PROBES_V3] + [p for p, _ in probes.CODE_EXACT] + [p for _, p, _ in probes.MULTI]
    held += [c["messages"][-1]["content"] for c in evaluate.cases() if c["group"] != "regression"]
    held += [p for p, _ in evaluate_repair.PROBES]
    return {normalized(p) for p in held}


# Filler and question words; the remaining words carry a prompt's content.
STOPWORDS = set("""a an the to in of for on with using by and or is are was be do does did can could would will should
you u your ur me my i im it its that this some so ok okay hey hi yo um uh hmm please pls plz kelus just what whats how
show tell give write make need want explain about there here now again lol quick question btw also alright""".split())


def content(words):
    return frozenset(w for w in words if w not in STOPWORDS)


def build(seed=303):
    rng = random.Random(seed)
    held = forbidden()
    multiword = [f" {h} " for h in held if " " in h]
    # One-word content ("hello" in "hello hello") would erase a whole basic
    # intent, so content matching needs at least two content words.
    held_content = {c for c in (content(h.split()) for h in held) if len(c) >= 2}
    removed = []

    def allowed(prompt):
        """Drop exact probes, prompts containing a probe, and prompts that ask
        for the same content words as a probe, give or take one word."""
        n = normalized(prompt)
        words = content(n.split())
        ok = bool(n) and n not in held and not any(h in f" {n} " for h in multiword)
        for probe in held_content if ok and words else ():
            smaller, larger = sorted((words, probe), key=len)
            if words == probe or (len(smaller) >= 3 and smaller <= larger and len(larger) - len(smaller) <= 1):
                ok = False
                break
        if not ok:
            removed.append(prompt)
        return ok

    pools = []  # (prompt variants, answer, weight group)
    for answer, cores in CHAT.items():
        pools.append((variants(cores, rng, 220), answer, "chat"))
    examples = {}
    for topic, explanation, example, single, plural, mass in CONCEPTS:
        cores = concept_cores(single, plural, mass)
        pools.append((variants(rng.sample(cores, len(cores)), rng, 160), explanation, "concept"))
        examples[explanation] = example
    unknown = list(dict.fromkeys(t.format(x=x, L=rng.choice(PYTHON + JAVASCRIPT)) for x in UNKNOWN_TARGETS for t in UNKNOWN_TEMPLATES))
    pools.append(([noisy(c, rng, wrap=rng.random() < 0.4) for c in rng.sample(unknown, 450)], UNKNOWN_PROGRAM, "chat"))
    for languages, snippets, fence in ((PYTHON, SNIPPETS, "python"), (JAVASCRIPT, JS_SNIPPETS, "javascript")):
        for name, code, *forms in snippets:
            cores = code_cores(languages, *forms, no_language=fence == "python")
            cores = rng.sample(cores, min(len(cores), 260))
            pools.append(([noisy(c, rng, wrap=rng.random() < 0.4) for c in cores] + cores[:40], f"```{fence}\n{code}\n```", "code"))

    train, validation = [], []
    for prompts, answer, _ in pools:
        prompts = [p for p in dict.fromkeys(prompts) if allowed(p)]
        rng.shuffle(prompts)
        cut = max(2, len(prompts) // 25)
        validation += [(p, answer) for p in prompts[:cut]]
        train += [(p, answer) for p in prompts[cut:]]
    # A prompt string never appears in both splits, even with different noise.
    validation_keys = {normalized(p) for p, _ in validation}
    train = [(p, a) for p, a in train if normalized(p) not in validation_keys]

    def turn(prompt, answer, supervise=True):
        reply = {"role": "assistant", "content": answer}
        if not supervise:
            reply["supervise"] = False
        return [{"role": "user", "content": prompt}, reply]

    records = [{"split": "train", "messages": turn(p, a)} for p, a in train]
    records += [{"split": "validation", "messages": turn(p, a)} for p, a in validation]
    # One earlier exchange, matching the runtime: the new turn must not copy it.
    for split, pairs, count in (("train", train, 7000), ("validation", validation, 300)):
        for _ in range(count):
            first, second = rng.choice(train), rng.choice(pairs)
            records.append({"split": split, "messages": turn(*first) + turn(*second)})
    # Short chat after any earlier exchange: the new message decides the reply,
    # e.g. "hi!" after "What would you like me to explain?" is still a greeting.
    chat_answers = {answer for _, answer, group in pools if group == "chat"}
    chat_pairs = [(p, a) for p, a in train if a in chat_answers]
    for _ in range(6000):
        records.append({"split": "train", "messages": turn(*rng.choice(train)) + turn(*rng.choice(chat_pairs))})
    # A vague request answered with a question, then the specific request.
    help_pairs = [(p, a) for p, a in train if a == HELP]
    requests = [(p, a) for p, a in train if a.startswith("```") or a == UNKNOWN_PROGRAM]
    for _ in range(1500):
        records.append({"split": "train", "messages": turn(*rng.choice(help_pairs)) + turn(*rng.choice(requests))})
    # Concept follow-ups: the example depends on the earlier concept.
    concept_pairs = [(p, a) for p, a in train if a in examples]
    follow_train = [f for f in FOLLOW_UPS if allowed(f)]
    follow_train = [noisy(f, rng, wrap=False) for f in follow_train for _ in range(3)] + follow_train
    for _ in range(2000):
        p, a = rng.choice(concept_pairs)
        records.append({"split": "train", "messages": turn(p, a) + turn(rng.choice(follow_train), examples[a])})
    validation_concepts = [(p, a) for p, a in validation if a in examples]
    for _ in range(60):
        p, a = rng.choice(validation_concepts)
        records.append({"split": "validation", "messages": turn(p, a) + turn(rng.choice(FOLLOW_UPS[:6]), examples[a])})
    # Corrections: an unrelated earlier reply is context only, never a target.
    answers = list(dict.fromkeys(a for _, a in train))
    corrections = [(p, a) for p, a in train if a == CORRECTION]
    for _ in range(1200):
        p, a = rng.choice(train)
        wrong = rng.choice([x for x in answers if x != a])
        fix_prompt, _ = rng.choice(corrections)
        records.append({"split": "train", "messages": turn(p, wrong, supervise=False) + turn(fix_prompt, CORRECTION)})

    unique = {json.dumps(r["messages"], sort_keys=True): r for r in records}
    return list(unique.values()), removed


if __name__ == "__main__":
    rows, removed = build()
    destination = Path(__file__).with_name("conversations-v3.jsonl")
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation")}
    print(json.dumps({"path": str(destination), "counts": counts, "decontaminated_prompts": len(removed)}))
