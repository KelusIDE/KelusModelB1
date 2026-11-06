"""Score everyday skills on held-out examples of data/everyday-v1.jsonl (data/make_everyday.py).

    python3 -m kelus.everyday_eval --checkpoint checkpoints/kani-otter-4-1.npz --out reports/everyday.json

  - recipes and learning: "How do I make bobozi?" and "teach me about optics" are searched for ("SEARCH: bobozi recipe",
    "SEARCH: optics"); the app then shows a page's recipe or answers from the results (dishes and topics never
    trained on). Food chat ("I love pancakes!") must stay chat.
  - phrases: "How do you say ... in Korean?" answered with exactly the reference translation, and answered in the
    right form ("In Korean: ...") at all; "What does ... mean?" with the exact English
  - Dolly (brainstorming, writing, questions about a passage): the replies are kept for reading; no single right answer
"""
import argparse
import json
from pathlib import Path
import time

from .model import Model

FAMILIES = ('everyday_recipe_search', 'everyday_learn_search', 'everyday_food_chat', 'everyday_phrases', 'everyday_dolly')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/everyday-v1.jsonl')
    parser.add_argument('--per-family', type=int, default=60)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [r for r in rows if r['split'] == 'validation']
    chosen = [r for family in FAMILIES for r in [x for x in rows if x['family'] == family][:args.per_family * (3 if family == 'everyday_phrases' else 1)]]
    results, started = [], time.monotonic()
    for row in chosen:
        reference = row['messages'][-1]['content']
        try:
            reply = ''.join(model.generate(row['messages'][:-1], max_tokens=min(400, model.config.context // 3), temperature=0)).strip()
        except ValueError:
            continue
        result = {'family': row['family'], 'language': row['language'], 'prompt': row['messages'][-2]['content'][:200], 'reply': reply[:800],
                  'reference': reference[:400], 'exact': reply == reference}
        result['searched'] = reply.startswith('SEARCH: ')
        if row['family'] == 'everyday_recipe_search':
            result['searched'] = result['searched'] and 'recipe' in reply.lower()
        if row['family'] == 'everyday_phrases':
            result['form'] = reply.split(':', 1)[0] == reference.split(':', 1)[0] if reference.startswith('In ') else reply.startswith('It means')
        results.append(result)
    pick = lambda family: [r for r in results if r['family'] == family]
    recipes, learning, food, phrases = (pick(f) for f in ('everyday_recipe_search', 'everyday_learn_search', 'everyday_food_chat', 'everyday_phrases'))
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'),
              'recipes_searched': f"{sum(r['searched'] for r in recipes)}/{len(recipes)}",
              'learning_searched': f"{sum(r['searched'] for r in learning)}/{len(learning)}",
              'food_chat_kept': f"{sum(not r['searched'] for r in food)}/{len(food)}",
              'phrases_exact': f"{sum(r['exact'] for r in phrases)}/{len(phrases)}", 'phrases_right_form': f"{sum(r['form'] for r in phrases)}/{len(phrases)}",
              'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
