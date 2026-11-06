"""Record direct-model baseline/candidate outputs on the same fixed prompts; never use app helpers."""
import argparse
import json
from pathlib import Path
import time

import torch
from kaggle.kani_instruction import amp, load_base, sha
from kelus.tokenizer import EOS, SPECIAL_IDS, conversation


def normalized(value):
    return ' '.join(value.strip().lower().rstrip('.!?').split())


def probe(checkpoint, cases, out, device='cuda', max_tokens=192):
    net, tokenizer, header = load_base(checkpoint)
    device = torch.device(device)
    net.to(device).eval()
    results = []
    for case in json.loads(Path(cases).read_text()):
        messages = case.get('messages') or [{'role': 'user', 'content': case['prompt']}]
        ids, _ = conversation(messages, generation=True, tokenizer=tokenizer)
        generated, started = [], time.monotonic()
        with torch.no_grad():
            for _ in range(min(max_tokens, net.context - len(ids))):
                with amp(device):
                    logits = net(torch.tensor([ids], device=device))[:, -1, :].float()
                for token in SPECIAL_IDS:
                    if token != EOS:
                        logits[:, token] = -torch.inf
                token = int(logits.argmax(-1).item())
                if token == EOS:
                    break
                ids.append(token); generated.append(token)
        answer = tokenizer.decode(generated)
        passed = None
        if case['kind'] == 'exact':
            passed = any(normalized(answer) == normalized(value) for value in case.get('accepted', [case['expected']]))
        elif case['kind'] == 'json':
            try:
                passed = json.loads(answer) == case['expected']
            except ValueError:
                passed = False
        results.append(dict(**case, response=answer, passed=passed, elapsed_seconds=round(time.monotonic() - started, 2)))
        automatic = [r for r in results if r['passed'] is not None]
        report = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), cases_sha256=sha(cases),
                      name=header.get('metadata', {}).get('display_name'), automatic_passed=sum(r['passed'] for r in automatic),
                      automatic_total=len(automatic), manual_ungraded=sum(r['passed'] is None for r in results), results=results,
                      limitation='Small task-specific probes, not an age/IQ/general-intelligence score. Manual/code tasks require review and execution.')
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix('.tmp.json'); temp.write_text(json.dumps(report, indent=2) + '\n'); temp.replace(out)
        print(json.dumps({'case': case['id'], 'passed': passed, 'response': answer}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=Path('evaluation/kani_instruction_probes.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default='cuda')
    parser.add_argument('--max-tokens', type=int, default=192)
    args = parser.parse_args()
    probe(args.checkpoint, args.cases, args.out, args.device, args.max_tokens)
