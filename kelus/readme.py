"""Refresh README.md's generated sections and figures from committed reports.

    python3 -m kelus.readme          # snapshot local foundation progress, then rewrite
    python3 -m kelus.readme --check  # exit 1 if README.md or a figure is stale; write nothing

Scores come only from reports that evaluated the exact checkpoint exported to
site/model/b1.json, so the README cannot describe a different model than the
one deployed. Foundation progress lives in gitignored training-runs/; running
this on the training machine copies a path-free summary to
reports/foundation-progress.json, which GitHub Actions can then render.
Standard library only, so CI needs no NumPy.
"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parent.parent
SITE = 'https://kelus-b1.web.app'
ASSETS = 'assets/readme'
SNAPSHOT = 'reports/foundation-progress.json'
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
MONO_WIDTH = 0.61  # advance per character, in em; slightly generous so reveals never clip
TILE, TILE_INK, BADGE_LABEL = '#bce5d1', '#1a3025', '151b18'
# Series greens validated against each surface (lightness band, chroma, 3:1 contrast).
THEMES = {
    'light': {'surface': '#fbfcfb', 'border': '#dfe7e2', 'ink': '#0f1a14', 'secondary': '#46564d',
              'muted': '#66726c', 'grid': '#e3e9e5', 'axis': '#c3cdc7', 'series': '#1f8a5b'},
    'dark': {'surface': '#151b18', 'border': '#26322c', 'ink': '#e8f1ec', 'secondary': '#b4c3bb',
             'muted': '#83918a', 'grid': '#232d28', 'axis': '#34423b', 'series': '#2fa874'},
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def esc(text):
    return html.escape(str(text), quote=True)


def fmt(n):
    return f'{n:,}'


def compact(n):
    for unit, size in (('M', 1e6), ('K', 1e3)):
        if n >= size:
            return f'{n / size:.3g}{unit}'
    return str(n)


def rate(score):
    return round(100 * score['passed'] / score['total']) if score['total'] else 0


def when(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def snapshot_foundation(root=ROOT):
    """Copy local foundation progress into a committed report, without PIDs or local paths."""
    run = root / 'training-runs/b1-foundation'
    if not (run / 'status.json').exists():
        return False
    rows = [read(run / 'status.json')]
    if (run / 'metrics.jsonl').exists():
        for line in (run / 'metrics.jsonl').read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # blank, or a line the trainer is still appending
    latest = max(rows, key=lambda row: row.get('updated_at') or 0)
    history = {row['step']: round(row['validation_loss'], 4) for row in sorted(rows[1:], key=lambda row: row.get('updated_at') or 0)
               if row.get('validation_loss') is not None}
    value = {key: latest.get(key) for key in ['stage', 'state', 'step', 'total_steps', 'parameters', 'context',
                                               'tokens_seen', 'validation_loss', 'eta_seconds']}
    value['updated_at'] = int(latest.get('updated_at') or 0)
    value['validation_history'] = [[step, history[step]] for step in sorted(history)]
    release = root / 'training-runs/auto-deploy/status.json'
    value['release_state'] = read(release).get('state') if release.exists() else None
    path = root / SNAPSHOT
    text = json.dumps(value, indent=2) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return False
    path.write_text(text, encoding='utf-8')
    return True


def released(root=ROOT):
    """The exported browser model plus the reports that evaluated that exact checkpoint."""
    model = read(root / 'site/model/b1.json')
    sha = model['source_checkpoint_sha256']

    def report(pattern, shape):
        for path in sorted(root.glob(pattern)):
            value = read(path)
            if value.get('checkpoint_sha256', value.get('sha256')) == sha and shape in value:
                return value
        raise ValueError(f'No {pattern} report with "{shape}" for the released checkpoint {sha[:12]}. '
                         'Evaluate that checkpoint before refreshing the README.')
    snapshot = root / SNAPSHOT
    return {'model': model,
            'benchmark': report('reports/benchmark/*.json', 'groups'),
            'original': report('reports/benchmark/*.json', 'scores'),
            'readiness': report('reports/code-readiness-*.json', 'python_total'),
            'foundation': read(snapshot) if snapshot.exists() else None}


def score_groups(data):
    benchmark, original, readiness = data['benchmark'], data['original'], data['readiness']
    groups, web = benchmark['groups'], readiness['website']['structure_checks']
    return [
        ('Held-out wording, practiced topics', [
            ('Chat replies', groups['chat']), ('Concept explanations', groups['concept']),
            ('Code requests', groups['code']), ('Two-turn chats', groups['multi_turn']),
            ('Earlier wording suite', original['scores']['heldout'])]),
        ('Regression checks', [
            ('Context after earlier turns', benchmark['context_regression']),
            ('Reported bug regressions', original['scores']['regression'])]),
        ('New tasks', [
            ('New Python specifications', {'passed': readiness['python_passed'], 'total': readiness['python_total']}),
            ('Website structure checks', {'passed': sum(v is True for v in web.values()), 'total': len(web)})]),
    ]


def samples(benchmark, limit=3):
    """Short passing replies with distinct intents, replayed in the hero."""
    picked, intents, replies = [], set(), set()
    for result in benchmark['results']:
        prompt, reply = result['prompt'], result['reply']
        if (result['passed'] and result['intent'] not in intents and reply not in replies
                and '\n' not in reply and len(reply) <= 56 and len(prompt) <= 40):
            picked.append(result)
            intents.add(result['intent'])
            replies.add(reply)
        if len(picked) == limit:
            break
    return picked


def foundation_done(f):
    return f['state'] == 'complete' and f['step'] >= f['total_steps']


def foundation_rate(f):
    return 100 if foundation_done(f) else min(99, int(100 * f['step'] / f['total_steps']))


def frame(width, height, theme, label, body, style=''):
    t = THEMES[theme]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{esc(label)}">
<title>{esc(label)}</title>
<style>
text {{ font-family: {SANS}; fill: {t['ink']}; }}
.mono {{ font-family: {MONO}; }}
.muted {{ fill: {t['muted']}; }}
.secondary {{ fill: {t['secondary']}; }}
.halo {{ paint-order: stroke; stroke: {t['surface']}; stroke-width: 4px; stroke-linejoin: round; }}
@keyframes fade {{ from {{ opacity: 0; }} }}
{style}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="16" fill="{t['surface']}" stroke="{t['border']}"/>
{body}
</svg>
'''


def hero_svg(data, theme):
    t, f = THEMES[theme], data['foundation']
    shown = samples(data['benchmark'])
    size, cw, x, prefix = 18, 18 * MONO_WIDTH, 248, 6
    tx = x + prefix * cw
    slot = 6
    period = slot * max(1, len(shown))
    share = 100 / max(1, len(shown))
    prompt_end, reply_start, reply_end, visible = (round(share * k, 2) for k in (0.2, 0.3, 0.55, 0.95))
    style = [f'''.sample:not(.s0) {{ opacity: 0; }}
.cover {{ fill: {t['surface']}; transform-box: fill-box; transform-origin: right; transform: scaleX(0); }}
.sample {{ animation: slot {period}s linear infinite; }}
.prompt {{ animation: prompt {period}s infinite; }}
.reply {{ animation: reply {period}s infinite; }}
.blink {{ animation: blink 1s linear infinite; }}
.dot {{ animation: pulse 2.4s ease-in-out infinite; }}
.ring {{ animation: breathe 4s ease-in-out infinite; }}
@keyframes slot {{ 0%, {visible}% {{ opacity: 1; }} {visible + 0.01}%, 100% {{ opacity: 0; }} }}
@keyframes prompt {{ 0% {{ transform: scaleX(1); }} {prompt_end}%, 100% {{ transform: scaleX(0); }} }}
@keyframes reply {{ 0%, {reply_start}% {{ transform: scaleX(1); }} {reply_end}%, 100% {{ transform: scaleX(0); }} }}
@keyframes blink {{ 0%, 49% {{ opacity: 1; }} 50%, 100% {{ opacity: 0; }} }}
@keyframes pulse {{ 0%, 100% {{ opacity: 0.25; }} 50% {{ opacity: 0.9; }} }}
@keyframes breathe {{ 0%, 100% {{ opacity: 0.15; }} 50% {{ opacity: 0.6; }} }}''']
    body = [f'''<rect class="ring" x="40" y="48" width="168" height="168" rx="40" fill="none" stroke="{t['series']}" stroke-width="2"/>
<rect x="48" y="56" width="152" height="152" rx="34" fill="{TILE}"/>
<text x="124" y="168" text-anchor="middle" font-size="104" style="font-family: Georgia, 'Times New Roman', serif; fill: {TILE_INK};">K</text>
<text class="mono muted" x="{x}" y="72" font-size="13" letter-spacing="1.5">RECORDED B1 REPLIES · HELD-OUT PROBES · TEMPERATURE 0</text>''']
    for column in range(6):
        for row in range(3):
            body.append(f'<circle class="dot" cx="{876 + column * 14}" cy="{66 + row * 14}" r="2.5" fill="{t["series"]}" '
                        f'style="animation-delay: {(column + row) * 0.15:.2f}s"/>')
    for i, result in enumerate(shown):
        delay = -((len(shown) - i) % len(shown)) * slot
        prompt, reply = result['prompt'], result['reply']
        width = len(reply) * cw
        style.append(f'''.s{i}, .s{i} .prompt, .s{i} .reply, .s{i} .cursor {{ animation-delay: {delay}s; }}
.s{i} .prompt {{ animation-timing-function: steps({len(prompt)}, end); }}
.s{i} .reply, .s{i} .cursor {{ animation-timing-function: steps({len(reply)}, end); }}
.s{i} .cursor {{ transform: translateX({width:.1f}px); animation-name: cursor{i}; animation-duration: {period}s; animation-iteration-count: infinite; }}
@keyframes cursor{i} {{ 0%, {reply_start - 0.01}% {{ opacity: 0; transform: translateX(0); }} {reply_start}% {{ opacity: 1; transform: translateX(0); }} {reply_end}%, 100% {{ opacity: 1; transform: translateX({width:.1f}px); }} }}''')
        body.append(f'''<g class="sample s{i}">
<text class="mono muted" x="{x}" y="118" font-size="{size}">you ›</text>
<text class="mono" x="{tx:.1f}" y="118" font-size="{size}">{esc(prompt)}</text>
<rect class="cover prompt" x="{tx - 1:.1f}" y="100" width="{len(prompt) * cw + 3:.1f}" height="24"/>
<text class="mono muted" x="{x}" y="154" font-size="{size}">b1  ›</text>
<text class="mono" x="{tx:.1f}" y="154" font-size="{size}">{esc(reply)}</text>
<rect class="cover reply" x="{tx - 1:.1f}" y="136" width="{width + 3:.1f}" height="24"/>
<g class="cursor"><rect class="blink" x="{tx + 2:.1f}" y="140" width="9" height="18" fill="{t['series']}"/></g>
</g>''')
    chips = [f"{fmt(data['model']['parameters'])} parameters",
             '{passed}/{total} held-out'.format(**data['benchmark']['overall'])]
    if f:
        chips.append('foundation ' + ('complete' if foundation_done(f) else f'{foundation_rate(f)}%'))
    left = x
    for chip in chips:
        width = len(chip) * 14 * MONO_WIDTH + 28
        body.append(f'''<rect x="{left}" y="186" width="{width:.1f}" height="32" rx="16" fill="{t['series']}" fill-opacity="0.12" stroke="{t['series']}" stroke-opacity="0.45"/>
<text class="mono secondary" x="{left + 14}" y="207" font-size="14">{esc(chip)}</text>''')
        left += width + 12
    return frame(1000, 264, theme, hero_alt(data), '\n'.join(body), '\n'.join(style))


def hero_alt(data):
    parts = [f"Kelus Model B1: {fmt(data['model']['parameters'])} parameters",
             '{passed}/{total} held-out probes'.format(**data['benchmark']['overall'])]
    if data['foundation']:
        parts.append(f"foundation pretraining {foundation_rate(data['foundation'])}%")
    return ', '.join(parts) + '. Animation replays recorded B1 replies to held-out prompts.'


def bar(x, y, width, height, radius=4):
    """Bar with a rounded data end and a square baseline end."""
    r = min(radius, width, height / 2)
    return (f'M{x},{y} H{x + width - r:.1f} Q{x + width:.1f},{y} {x + width:.1f},{y + r} '
            f'V{y + height - r} Q{x + width:.1f},{y + height} {x + width - r:.1f},{y + height} H{x} Z')


def results_svg(data, theme):
    t = THEMES[theme]
    left, x0, x1, row = 40, 320, 820, 34
    checkpoint = Path(data['benchmark']['checkpoint']).name.removesuffix('.npz')
    body, bars = [], []
    y, top, index = 96, None, 0
    for title, items in score_groups(data):
        y += 26
        body.append(f'<text class="muted" x="{left}" y="{y}" font-size="12" letter-spacing="1">{esc(title.upper())}</text>')
        y += 10
        top = top or y  # gridlines start below the first group header
        for label, score in items:
            mid = y + row / 2
            width = (x1 - x0) * score['passed'] / score['total'] if score['total'] else 0
            body.append(f'<text class="secondary" x="{left}" y="{mid + 5}" font-size="14">{esc(label)}</text>')
            if width:
                bars.append(f'<path class="bar" d="{bar(x0, mid - 8, width, 16)}" fill="{t["series"]}" style="animation-delay: {index * 0.06:.2f}s"/>')
            body.append(f'<text class="value halo" x="{x0 + width + 10:.1f}" y="{mid + 5}" font-size="14" '
                        f'style="animation-delay: {0.5 + index * 0.06:.2f}s">{score["passed"]}/{score["total"]} · {rate(score)}%</text>')
            y += row
            index += 1
    bottom = y + 6
    grid = [f'<line x1="{x0 + (x1 - x0) * k / 4}" y1="{top}" x2="{x0 + (x1 - x0) * k / 4}" y2="{bottom}" stroke="{t["grid"]}" stroke-width="1"/>'
            f'<text class="muted" x="{x0 + (x1 - x0) * k / 4}" y="{bottom + 20}" font-size="12" text-anchor="middle">{25 * k}%</text>'
            for k in range(1, 5)]
    grid.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>'
                f'<text class="muted" x="{x0}" y="{bottom + 20}" font-size="12" text-anchor="middle">0%</text>')
    header = (f'<text x="{left}" y="44" font-size="18" font-weight="600">Released model evaluation</text>'
              f'<text class="muted" x="{left}" y="68" font-size="13">{esc(checkpoint)} · {fmt(data["model"]["parameters"])} parameters · '
              'temperature 0 · share of checks passed</text>')
    style = ('.bar { transform-box: fill-box; transform-origin: left; animation: grow 0.9s cubic-bezier(0.2, 0.7, 0.2, 1) backwards; }\n'
             '.value { animation: fade 0.4s ease-out backwards; }\n'
             '@keyframes grow { from { transform: scaleX(0); } }')
    label = 'Released model evaluation: ' + '; '.join(f'{name} {s["passed"]} of {s["total"]}'
                                                       for _, items in score_groups(data) for name, s in items)
    return frame(1000, bottom + 44, theme, label, '\n'.join([header] + grid + bars + body), style)


def nice_ceiling(value):
    for step in (0.5, 1, 2, 5, 10, 20):
        if value / step <= 5:
            return step * -(-value // step), step
    return value, value


def foundation_svg(f, theme):
    t = THEMES[theme]
    left, right, px0, py0, py1 = 40, 960, 88, 156, 316
    total, history = f['total_steps'], f['validation_history']
    done, percent = foundation_done(f), foundation_rate(f)
    top, step = nice_ceiling(max([loss for _, loss in history] or [1]))
    sx = lambda s: px0 + (right - px0) * s / total
    sy = lambda loss: py1 - (py1 - py0) * loss / top
    body = [f'<text x="{left}" y="44" font-size="18" font-weight="600">Foundation pretraining</text>',
            f'<text class="muted" x="{left}" y="68" font-size="13">{fmt(f["parameters"])} parameters · sampled validation loss, '
            f'nats per token (lower is better) · snapshot {when(f["updated_at"])}</text>',
            f'<text x="{left}" y="104" font-size="14" font-weight="600">Step {fmt(f["step"])} of {fmt(total)}</text>',
            f'<text class="secondary" x="{right}" y="104" font-size="14" text-anchor="end">{"complete" if done else f"{percent}%"}</text>',
            f'<rect x="{left}" y="114" width="{right - left}" height="8" rx="4" fill="{t["series"]}" fill-opacity="0.18"/>',
            f'<rect class="meter" x="{left}" y="114" width="{(right - left) * percent / 100:.1f}" height="8" rx="4" fill="{t["series"]}"/>']
    ticks = int(top / step)
    for k in range(ticks + 1):
        value, y = k * step, sy(k * step)
        body.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{t["axis"] if k == 0 else t["grid"]}" stroke-width="1"/>'
                    f'<text class="muted" x="{px0 - 10}" y="{y + 4:.1f}" font-size="12" text-anchor="end">{value:g}</text>')
    for k in range(5):
        s = total * k / 4
        body.append(f'<text class="muted" x="{sx(s):.1f}" y="{py1 + 22}" font-size="12" text-anchor="middle">{compact(int(s))}</text>')
    body.append(f'<text class="muted" x="{right}" y="{py1 + 40}" font-size="12" text-anchor="end">optimizer step</text>')
    if history:
        points = ' '.join(f'{sx(s):.1f},{sy(loss):.1f}' for s, loss in history)
        last_s, last_loss = history[-1]
        body += [f'<path class="area" d="M{sx(history[0][0]):.1f},{py1} L{points} L{sx(last_s):.1f},{py1} Z" fill="{t["series"]}" fill-opacity="0.1"/>',
                 f'<path class="line" pathLength="1" d="M{points}" fill="none" stroke="{t["series"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>',
                 f'<circle class="end" cx="{sx(last_s):.1f}" cy="{sy(last_loss):.1f}" r="5" fill="{t["series"]}" stroke="{t["surface"]}" stroke-width="2"/>',
                 f'<text class="end halo" x="{sx(last_s) + 10:.1f}" y="{sy(last_loss) - 10:.1f}" font-size="13" font-weight="600">{last_loss:.3f}</text>']
    style = ('.meter { transform-box: fill-box; transform-origin: left; animation: grow 1s cubic-bezier(0.2, 0.7, 0.2, 1) backwards; }\n'
             '.line { stroke-dasharray: 1; animation: draw 1.6s ease-out backwards; }\n'
             '.area { animation: fade 0.8s ease-out 1.2s backwards; }\n'
             '.end { animation: fade 0.3s ease-out 1.5s backwards; }\n'
             '@keyframes grow { from { transform: scaleX(0); } }\n'
             '@keyframes draw { from { stroke-dashoffset: 1; } }')
    return frame(1000, 368, theme, foundation_alt(f), '\n'.join(body), style)


def foundation_alt(f):
    loss = f'; validation loss {f["validation_loss"]:.3f}' if f.get('validation_loss') is not None else ''
    return f'Foundation pretraining at step {fmt(f["step"])} of {fmt(f["total_steps"])}{loss}.'


def picture(name, alt):
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{ASSETS}/{name}-dark.svg">\n'
            f'  <img src="{ASSETS}/{name}-light.svg" width="100%" alt="{esc(alt)}">\n</picture>')


def badge(label, message, color, link, alt):
    part = lambda text: quote(text.replace('-', '--').replace('_', '__'), safe='')
    src = f'https://img.shields.io/badge/{part(label)}-{part(message)}-{color}?style=flat-square&labelColor={BADGE_LABEL}'
    return f'  <a href="{link}"><img src="{esc(src)}" alt="{esc(alt)}"></a>'


def badge_color(score):
    return 'bce5d1' if rate(score) >= 90 else 'fab219' if rate(score) >= 50 else 'e66767'


def badges(data):
    live = {'style': 'flat-square', 'labelColor': BADGE_LABEL}
    status = 'https://img.shields.io/website?' + urlencode({'url': SITE, 'label': 'kelus-b1.web.app', 'up_message': 'online',
                                                            'down_message': 'offline', 'up_color': '2fa874'} | live)
    deployed = 'https://img.shields.io/badge/dynamic/json?' + urlencode({'url': SITE + '/model/b1.json', 'query': '$.parameters',
                                                                         'label': 'deployed model', 'suffix': ' parameters', 'color': 'bce5d1'} | live)
    overall, readiness = data['benchmark']['overall'], data['readiness']
    new = {'passed': readiness['python_passed'], 'total': readiness['python_total']}
    lines = ['<p align="center">',
             f'  <a href="{SITE}"><img src="{esc(status)}" alt="Web app status"></a>',
             f'  <a href="{SITE}/model/b1.json"><img src="{esc(deployed)}" alt="Parameters in the deployed model"></a>',
             badge('held-out probes', '{passed}/{total}'.format(**overall), badge_color(overall), '#results-and-roadmap',
                   'Held-out probes: {passed} of {total}'.format(**overall)),
             badge('new programs', '{passed}/{total}'.format(**new), badge_color(new), '#results-and-roadmap',
                   'New program specifications: {passed} of {total}'.format(**new))]
    f = data['foundation']
    if f:
        message = 'complete' if foundation_done(f) else f'{foundation_rate(f)}%'
        lines.append(badge('foundation pretraining', message, 'bce5d1', '#foundation-training', 'Foundation pretraining: ' + message))
    return '\n'.join(lines + ['</p>'])


def stage(f):
    if f is None or not foundation_done(f):
        return 0
    return {'held': 2, 'deploying': 3, 'deployed': 4}.get(f.get('release_state'), 1)


def pipeline(f):
    current = stage(f)
    nodes = [('F', 'Foundation pretraining'), ('I', 'Instruction training'),
             ('G', 'Release gates and<br/>human verification'), ('D', 'Auto-deploy to<br/>kelus-b1.web.app')]
    lines = ['flowchart LR']
    for i, (key, label) in enumerate(nodes):
        status = 'done' if i < current else 'now' if i == current else 'waiting'
        note = {'done': '✓ done', 'now': '● in progress', 'waiting': 'waiting'}[status]
        if i == 0 and status == 'now':
            note += f' · step {fmt(f["step"])} / {fmt(f["total_steps"])}' if f else ''
        lines.append(f'    {key}["{label}<br/>{note}"]:::{status}')
    lines += ['    F --> I --> G --> D',
              f'    classDef done fill:{TILE},stroke:{THEMES["light"]["series"]},color:{TILE_INK}',
              f'    classDef now fill:{TILE_INK},stroke:{THEMES["dark"]["series"]},stroke-width:3px,color:{THEMES["dark"]["ink"]}',
              f'    classDef waiting fill:none,stroke:{THEMES["dark"]["muted"]},stroke-dasharray:4 3,color:{THEMES["dark"]["muted"]}']
    return '\n'.join(lines)


def foundation_section(f):
    if f is None:
        return 'No foundation snapshot yet. On the training machine, run `python3 -m kelus.readme` to record one.'
    done = foundation_done(f)
    state = 'complete' if done else f['state'].replace('_', ' ')
    summary = (f'**Pretraining {state}: step {fmt(f["step"])} of {fmt(f["total_steps"])} ({foundation_rate(f)}%).** '
               f'{compact(f["tokens_seen"])} tokens seen')
    if f.get('validation_loss') is not None:
        summary += f'; sampled validation loss {f["validation_loss"]:.3f} nats per token'
    summary += f'. Snapshot taken {when(f["updated_at"])}'
    summary += (f'; about {f["eta_seconds"] / 3600:.1f} hours of this phase remained then.'
                if f['state'] == 'running' and f.get('eta_seconds') else '.')
    return '\n'.join([
        picture('foundation', foundation_alt(f)), '', summary,
        'Loss measures next-token prediction on held-out documents, not coding ability. Progress lives on the '
        'training machine; `python3 -m kelus.readme` copies it to [reports/foundation-progress.json](reports/foundation-progress.json).',
        '', '<details>', '<summary><strong>Release pipeline: where the next model stands</strong></summary>', '',
        '```mermaid', pipeline(f), '```', '',
        'Every stage must finish before the release watcher publishes. See [AUTO_DEPLOY.md](AUTO_DEPLOY.md).', '', '</details>'])


def results_section(data):
    overall, readiness = data['benchmark']['overall'], data['readiness']
    web = readiness['website']['structure_checks']
    checkpoint = Path(data['benchmark']['checkpoint']).name.removesuffix('.npz')
    rows = ['| Group | Check | Passed | Rate |', '| :--- | :--- | ---: | ---: |']
    for title, items in score_groups(data):
        rows += [f'| {title} | {label} | {s["passed"]}/{s["total"]} | {rate(s)}% |' for label, s in items]
    return '\n'.join([
        f'The released model (`{checkpoint}`, {fmt(data["model"]["parameters"])} parameters) scores '
        f'**{overall["passed"]}/{overall["total"]}** on held-out wordings of its practiced topics. That benchmark '
        f'measures familiar tasks: the separate new-program check scores **{readiness["python_passed"]}/{readiness["python_total"]}**, '
        f'and the website prompt passes **{sum(v is True for v in web.values())}/{len(web)}** of its requested structure checks.',
        '', picture('results', 'Bar chart of the released model\'s pass rates; the table below lists every value.'), '',
        '<details>', '<summary><strong>Score table</strong></summary>', '', *rows, '', '</details>', '',
        'Read [MODEL_CARD.md](MODEL_CARD.md) and [evaluation reports](reports/benchmark/) for the methods and limitations.'])


def sections(data):
    hero = '\n'.join(['<p align="center">', f'  <a href="{SITE}">', *('    ' + line for line in picture('hero', hero_alt(data)).splitlines()),
                      '  </a>', '</p>'])
    return {'hero': hero, 'badges': badges(data), 'foundation': foundation_section(data['foundation']),
            'results': results_section(data)}


def fill(text, generated):
    for name, body in generated.items():
        pattern = re.compile(rf'(<!-- readme:{name} -->\n).*?(<!-- /readme:{name} -->)', re.S)
        if not pattern.search(text):
            raise ValueError(f'README.md is missing the <!-- readme:{name} --> ... <!-- /readme:{name} --> markers.')
        text = pattern.sub(lambda m: m.group(1) + body.strip('\n') + '\n' + m.group(2), text, count=1)
    return text


def outputs(root=ROOT):
    data = released(root)
    files = {}
    for theme in THEMES:
        files[f'{ASSETS}/hero-{theme}.svg'] = hero_svg(data, theme)
        files[f'{ASSETS}/results-{theme}.svg'] = results_svg(data, theme)
        if data['foundation']:
            files[f'{ASSETS}/foundation-{theme}.svg'] = foundation_svg(data['foundation'], theme)
    files['README.md'] = fill((root / 'README.md').read_text(encoding='utf-8'), sections(data))
    return files


def refresh(root=ROOT, write=True):
    """Return the relative paths that differ from the generated output, writing them unless asked not to."""
    stale = []
    for name, text in outputs(root).items():
        path = root / name
        if path.exists() and path.read_text(encoding='utf-8') == text:
            continue
        stale.append(name)
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='exit 1 if anything is stale; write nothing')
    args = parser.parse_args()
    if not args.check and snapshot_foundation():
        print('updated:', SNAPSHOT)
    stale = refresh(write=not args.check)
    for name in stale:
        print('stale:' if args.check else 'updated:', name)
    if args.check and stale:
        sys.exit(1)
    if not stale:
        print('README.md and its figures are current.')


if __name__ == '__main__':
    main()
