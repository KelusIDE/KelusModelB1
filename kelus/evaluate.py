"""Small, disclosed chat/code regression suite. No generated code is executed."""
import argparse
import ast
import hashlib
import json
import operator
from pathlib import Path
import re
import time
from .model import Model


def user(text):
    return {"role": "user", "content": text}


def code_result(text, function, trials):
    """Interpret only a single return expression in a tiny AST whitelist.

    This never calls eval/exec or imports generated code. Attribute access,
    comprehensions, arbitrary calls, statements, and decorators are refused.
    """
    if "```" in text:
        parts = text.split("```")
        if len(parts) != 3 or parts[0].strip() or parts[2].strip():
            return False
        text = parts[1].removeprefix("python").strip()
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod}
    compare = {ast.Eq: operator.eq, ast.NotEq: operator.ne}
    allowed_calls = {"len": len, "sum": sum, "max": max, "min": min}

    def value(node, scope):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool, type(None)):
            if isinstance(node.value, (int, float)) and abs(node.value) > 1_000_000:
                raise ValueError("Constant exceeds evaluator limits")
            return node.value
        if isinstance(node, ast.Name):
            return scope[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = value(node.left, scope), value(node.right, scope)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError("Only numeric arithmetic is supported")
            result = binary[type(node.op)](left, right)
            if abs(result) > 1_000_000_000:
                raise ValueError("Result exceeds evaluator limits")
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand, scope)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in compare:
            return compare[type(node.ops[0])](value(node.left, scope), value(node.comparators[0], scope))
        if isinstance(node, ast.Subscript):
            return value(node.value, scope)[value(node.slice, scope)]
        if isinstance(node, ast.Slice):
            return slice(*(value(item, scope) if item else None for item in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in allowed_calls and len(node.args) == 1 and not node.keywords:
            return allowed_calls[node.func.id](value(node.args[0], scope))
        if isinstance(node, ast.JoinedStr):
            # Plain f-string fields only: no conversions or format specifications.
            parts = []
            for part in node.values:
                if isinstance(part, ast.FormattedValue):
                    item = value(part.value, scope)
                    if part.conversion != -1 or part.format_spec is not None or type(item) not in (str, int, float, bool):
                        raise ValueError("Unsupported f-string field")
                    parts.append(str(item))
                else:
                    parts.append(value(part, scope))
            return "".join(parts)
        raise ValueError("Outside the evaluator's expression subset")

    try:
        tree = ast.parse(text)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return False
        fn = tree.body[0]
        if fn.name != function or fn.decorator_list or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
            return False
        args = fn.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.posonlyargs:
            return False
        for inputs, expected in trials:
            if len(args.args) != len(inputs):
                return False
            scope = {arg.arg: item for arg, item in zip(args.args, inputs)}
            actual = value(fn.body[0].value, scope)
            if type(actual) is not type(expected) or actual != expected:
                return False
        return True
    except (SyntaxError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError):
        return False


SQUARE = "```python\ndef square(x):\n    return x * x\n```"


def cases():
    items = []

    def chat(name, prompt, words, group="heldout", history=None):
        items.append({"name": name, "group": group, "messages": (history or []) + [user(prompt)], "words": words})

    chat("hello punctuation", "hello!", ["hi", "hello"], "regression")
    chat("hi punctuation", "hi!", ["hi", "hello"], "regression")
    chat("clarification", "what", ["misunderstood", "clarify", "rephrase"], "regression")
    previous = [user("Write a Python square function."), {"role": "assistant", "content": SQUARE}]
    chat("hello after code", "hello!", ["hi", "hello"], "regression", previous)
    chat("repair after code", "what", ["misunderstood", "clarify", "rephrase"], "regression", previous)
    chat("new greeting wording", "hey there, Kelus!", ["hi", "hello"])
    chat("new introduction wording", "could you tell me who you are?", ["kelus"])
    chat("new thanks wording", "thank you, that was helpful!", ["welcome", "glad"])
    chat("new help wording", "could you help me write some code?", ["build", "language", "code"])
    chat("new variable question", "can you explain what a variable is?", ["name", "value"])
    chat("new loop question", "please explain what a loop does", ["repeat", "repeats"])
    chat("new debugging question", "my program is crashing, can you help?", ["error", "code"])
    chat("new clarification wording", "wait, I don't get it", ["misunderstood", "clarify", "rephrase"])
    chat("greeting in three-turn chat", "hey there, Kelus!", ["hi", "hello"], history=previous + [user("thanks"), {"role": "assistant", "content": "You're welcome!"}])

    for name, prompt, function, trials in [
        ("addition", "could you write a function in Python that adds two numbers?", "add", [((2, 3), 5), ((-4, 4), 0), ((0, 9), 9)]),
        ("square", "please show Python code for squaring a number", "square", [((3,), 9), ((-4,), 16), ((0,), 0)]),
        ("multiplication", "help me multiply two numbers using Python", "multiply", [((3, 4), 12), ((-2, 5), -10)]),
        ("even", "how can I check whether a number is even using Python?", "is_even", [((2,), True), ((3,), False), ((0,), True)]),
        ("reverse", "could you reverse a string using Python?", "reverse", [(("abc",), "cba"), (("",), ""), (("hello!",), "!olleh")]),
        ("sum", "please show a Python function that sums a list", "total", [(([1, 2, 3],), 6), (([],), 0), (([-2, 4],), 2)]),
    ]:
        items.append({"name": "new " + name + " code", "group": "heldout", "messages": [user(prompt)], "function": function, "trials": trials})
    for prompt in ("what is the capital of France?", "can you build a complete Minecraft client?", "write a Python web server with login and a database"):
        items.append({"name": "exploration: " + prompt, "group": "exploratory", "messages": [user(prompt)]})
    return items


def screenshot_sequence(model, temperature):
    """Replay the user's five messages, using the model's own prior replies."""
    history, results = [], []
    prompts = ["Write a Python square function.", "hello!", "what", "hi!", "hello"]
    for index, prompt in enumerate(prompts):
        history.append(user(prompt))
        reply = "".join(model.generate(history, max_tokens=180, temperature=temperature))
        if index == 0:
            passed = code_result(reply, "square", [((3,), 9), ((-4,), 16), ((0,), 0)])
        elif index == 2:
            passed = bool(re.search(r"\b(misunderstood|clarify|rephrase)\b", reply.lower()))
        else:
            passed = "```" not in reply and bool(re.search(r"\b(hi|hello)\b", reply.lower()))
        results.append({"prompt": prompt, "reply": reply, "passed": passed})
        history.append({"role": "assistant", "content": reply})
    return {"passed": sum(result["passed"] for result in results), "total": len(results), "turns": results}


def run(checkpoint, temperature=0, data="data/conversations.jsonl"):
    model, header, _ = Model.load(checkpoint)
    training_prompts = set()
    if Path(data).exists():
        for line in Path(data).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("split") == "train":
                training_prompts.update(message["content"] for message in record["messages"] if message["role"] == "user")
    results = []
    for case in cases():
        if case["group"] == "heldout" and case["messages"][-1]["content"] in training_prompts:
            raise ValueError(f"Evaluation prompt leaked into training: {case['name']}")
        started = time.monotonic()
        reply = "".join(model.generate(case["messages"], max_tokens=180, temperature=temperature))
        if "function" in case:
            passed = code_result(reply, case["function"], case["trials"])
        elif "words" in case:
            # This is a narrow keyword check, not a measure of general fluency.
            passed = (len(reply.split()) >= 3 and "```" not in reply and "\ufffd" not in reply and any(re.search(r"\b" + re.escape(word) + r"\b", reply.lower()) for word in case["words"]))
        else:
            passed = None
        results.append({**case, "reply": reply, "passed": passed, "seconds": round(time.monotonic() - started, 3)})
    scores = {}
    for group in ("regression", "heldout"):
        selected = [result for result in results if result["group"] == group]
        scores[group] = {"passed": sum(result["passed"] for result in selected), "total": len(selected)}
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "steps": header.get("metadata", {}).get("steps"), "parameters": model.parameter_count,
            "temperature": temperature, "scores": scores,
            "screenshot_sequence": screenshot_sequence(model, temperature),
            "scope": "Known bug regressions and unseen wording on practiced topics. Chat uses keyword checks; six Python functions use whitelisted AST interpretation. Exploratory prompts are unscored. Not a general intelligence or coding benchmark.",
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/b1-v3.best.npz")
    parser.add_argument("--out", default="reports/conversation-eval.json")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--data", default="data/conversations-v3.jsonl", help="Training file checked for held-out prompt leakage")
    args = parser.parse_args()
    result = run(args.checkpoint, args.temperature, args.data)
    destination = Path(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": args.checkpoint, "scores": result["scores"], "report": str(destination)}))
    for item in result["results"]:
        print(f"{item['passed']} | {item['name']} | {item['reply']!r}")


if __name__ == "__main__":
    main()
