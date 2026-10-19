"""Wider held-out benchmark: hand-written probes the curricula never template.

Chat probes require an exact canonical reply. Python function probes use the
bounded AST interpreter from `kelus.evaluate`; generated code is never run.
Every probe prompt is checked against the training split of each data file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from .model import Model
from .evaluate import code_result
from . import probes

TRIALS = {"square": [((3,), 9), ((-4,), 16), ((0,), 0)]}


def user(text):
    return {"role": "user", "content": text}


def cases():
    items = []
    for intent, prompts in probes.CHAT.items():
        items += [{"group": "chat", "intent": intent, "messages": [user(p)]} for p in prompts]
    for topic, prompts in probes.CONCEPT_PROBES.items():
        items += [{"group": "concept", "intent": "concept:" + topic, "messages": [user(p)]} for p in prompts]
    for prompt, function, trials in probes.CODE_PROBES + probes.CODE_PROBES_V3:
        items.append({"group": "code", "intent": "fn:" + function, "messages": [user(prompt)], "function": function, "trials": trials})
    for prompt, intent in probes.CODE_EXACT:
        items.append({"group": "code", "intent": intent, "messages": [user(prompt)]})
    for history, prompt, intent in probes.MULTI:
        messages = [m for p, a in history for m in (user(p), {"role": "assistant", "content": a})] + [user(prompt)]
        case = {"group": "multi_turn", "intent": intent, "messages": messages}
        if intent.startswith("code:") and intent[5:] in TRIALS:
            case.update(function=intent[5:], trials=TRIALS[intent[5:]])
        items.append(case)
    return items


def passed(case, reply):
    if "function" in case:
        return code_result(reply, case["function"], case["trials"])
    accepted = probes.INTENTS.get(case["intent"]) or probes.EXAMPLES[case["intent"]]
    return reply.strip() in accepted


def normalized(text):
    """Case and punctuation are ignored, so near-duplicates also count as leaks."""
    return re.sub(r"[^a-z0-9 ]", "", " ".join(text.casefold().split())).strip()


def training_prompts(files):
    seen = set()
    for file in files:
        for line in Path(file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("split", "train") == "train":
                    seen.update(normalized(m["content"]) for m in record["messages"] if m["role"] == "user")
    return seen


def run(checkpoint, data_files=(), max_tokens=120):
    model, header, _ = Model.load(checkpoint)
    leaked = training_prompts(data_files)
    results, started = [], time.monotonic()
    for case in cases():
        prompt = case["messages"][-1]["content"]
        if normalized(prompt) in leaked:
            raise ValueError(f"Probe prompt appears in training data: {prompt!r}")
        reply = "".join(model.generate(case["messages"], max_tokens=max_tokens, temperature=0))
        results.append({**case, "prompt": prompt, "reply": reply, "passed": bool(passed(case, reply))})
    groups = {}
    for result in results:
        group = groups.setdefault(result["group"], {"passed": 0, "total": 0})
        group["passed"] += result["passed"]; group["total"] += 1
    total = sum(r["passed"] for r in results)
    context = []
    for prior, answer in probes.CONTEXT_PRIORS:
        for follow_up, intent in probes.CONTEXT_FOLLOW_UPS:
            messages = [user(prior), {"role": "assistant", "content": answer}, user(follow_up)]
            reply = "".join(model.generate(messages, max_tokens=max_tokens, temperature=0))
            context.append({"prior": prior, "prompt": follow_up, "intent": intent, "reply": reply, "passed": reply.strip() in probes.INTENTS[intent]})
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "steps": header.get("metadata", {}).get("steps"), "parameters": model.parameter_count,
            "overall": {"passed": total, "total": len(results), "accuracy": round(total / len(results), 4)},
            "groups": groups, "seconds": round(time.monotonic() - started, 1),
            "context_regression": {"passed": sum(c["passed"] for c in context), "total": len(context),
                                   "scope": "Trained messages after each kind of earlier exchange; not held out.",
                                   "failures": [c for c in context if not c["passed"]]},
            "leak_checked_against": [str(f) for f in data_files],
            "scope": "Hand-written held-out wording on practiced topics. Exact canonical replies for chat; AST-checked Python functions. Not a general benchmark.",
            "results": [{k: r[k] for k in ("group", "intent", "prompt", "reply", "passed")} for r in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/b1-v3.best.npz")
    parser.add_argument("--data", nargs="*", default=["data/conversations.jsonl", "data/repair.jsonl", "data/repair-v2.jsonl", "data/conversations-v3.jsonl"], help="Training files checked for probe leakage")
    parser.add_argument("--out")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()
    report = run(args.checkpoint, [d for d in args.data if Path(d).exists()])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{k: report[k] for k in ("checkpoint", "steps", "overall", "groups", "seconds")},
                      "context_regression": {k: report["context_regression"][k] for k in ("passed", "total")}}))
    if args.show_failures:
        for r in report["results"]:
            if not r["passed"]:
                print(f"FAIL {r['intent']:<22} {r['prompt']!r} -> {r['reply'][:90]!r}")


if __name__ == "__main__":
    main()
