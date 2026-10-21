"""Generate a reproducible capability probe report; never execute model code.

Score with node cli/evaluate-capabilities.mjs REPORT.json. Most probes hold out
wording within trained families; novel topics are labeled separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .model import Model
from .runtime import prepare_reply

CASES = [
    ("pancakes", "chat", "Pancakes are my favorite breakfast.", ["pancakes"]),
    ("toppings", "chat", "I like pancakes with strawberries.", ["pancakes", "strawberries"]),
    ("hobby", "chat", "I am a big fan of drawing!", ["drawing"]),
    ("negative", "chat", "I do not like soup.", ["prefer", "soup"]),
    ("novel_food", "novel_chat", "I really enjoy omelettes.", ["omelette"]),
    ("novel_hobby", "novel_chat", "I love astronomy.", ["astronomy"]),
    ("circle", "svg", "Draw a small blue circle as SVG.", {"tag": "circle", "attrs": {"r": "32", "fill": "#3b82f6"}}),
    ("square", "svg", "Generate an SVG image showing a large purple square.", {"tag": "rect", "attrs": {"width": "176", "height": "176", "fill": "#a855f7"}}),
    ("heart", "svg", "Draw a medium pink heart as SVG.", {"tag": "path", "attrs": {"fill": "#ec4899"}, "contains": "scale(0.6364)"}),
    ("scene", "svg", "Draw a green house with a brown roof as SVG.", {"tag": "rect", "attrs": {"fill": "#22c55e"}, "contains": "#92400e"}),
    ("composition", "svg", "Draw a blue circle above of a green square as SVG.", {"tag":"circle","attrs":{"cx":"128","cy":"64","fill":"#3b82f6"},"contains": 'y="160"'}),
    ("lru", "code", "Build a Python LRUCache with get(key, default=None) and put(key, value). Reject capacity <= 0. Reads update recency, and insertion evicts the least recently used entry.", "lru"),
    ("group", "code", "Please implement group_by to group a list of dictionaries by a requested key without modifying the input. Use Python.", "group"),
    ("flatten", "code", "Please implement flatten to flatten nested lists of any depth while preserving item order. Use Python.", "flatten"),
    ("intervals", "code", "Please implement merge_ranges to merge overlapping intervals, sort the result, and preserve the original input. Use Python.", "intervals"),
    ("csv", "code", "Please implement parse_rows to parse CSV text with headers, quoted commas, and quoted newlines using the standard library. Use Python.", "csv"),
]
MEMORY = [
    {"role":"user","content":"I love pancakes."},
    {"role":"assistant","content":"Pancakes! What do you like most about pancakes?"},
    {"role":"user","content":"How are you?"},
    {"role":"assistant","content":"Ready to help. How is your day going?"},
    {"role":"user","content":"What do I like?"},
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    identity = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model, header, _ = Model.load(args.checkpoint)
    cases = list(CASES) + [("memory","chat",MEMORY,["pancakes"]),
        ("correction","chat",MEMORY[:2] + [{"role":"user","content":"No, I meant waffles."}],["waffles"])]
    rows = []
    for name, family, prompt, expected in cases:
        messages = prompt if isinstance(prompt,list) else [{"role":"user","content":prompt}]
        budget = 768 if family in ("code","svg") else 100
        start=time.monotonic()
        try:
            retained, omitted, tool = prepare_reply(model, messages, budget)
            reply = tool[0] if tool else "".join(model.generate(retained,max_tokens=budget,temperature=0))
            row={"id":name,"family":family,"messages":messages,"expected":expected,"reply":reply,"omitted":omitted}
        except ValueError as error:
            row={"id":name,"family":family,"messages":messages,"expected":expected,"reply":"","error":str(error)}
        row["seconds"]=round(time.monotonic()-start,3)
        rows.append(row)
        print(json.dumps({"id":name,"seconds":row["seconds"],"reply":row["reply"][:180]},ensure_ascii=False),flush=True)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != identity:
        raise RuntimeError("Checkpoint changed during evaluation. Retry with a frozen copy.")
    report={"checkpoint":str(args.checkpoint),"checkpoint_sha256":identity,
            "parameters":model.parameter_count,"training":header.get("metadata",{}),
            "scope":"Small diagnostic suite, mostly held-out wording within trained families. Novel topics reported separately. Not a general coding or reasoning benchmark.",
            "cases":rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")

if __name__ == "__main__":
    main()
