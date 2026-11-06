# Paragraph pilot review

Parent: completed Big Kani v2 chat, step 9000. The fixed development prompts
are in `evaluation/paragraph_v1.json`; raw before/after replies are saved in
`training-runs/kani-big-v2-paragraph-pilot/`. These are eight development
checks, not a general writing benchmark. Six require qualitative review.

## Baseline, recorded before the pilot

| Prompt | Observed behavior |
| --- | --- |
| Indoor cats and play | 54 words, repetitive claims; no example game |
| Rainy library scene | Generic description of sounds/smells; misses the requested arrival scene |
| Repair-club notes | Repeats instructions as bullets instead of rewriting the facts |
| School garden | Multiple paragraphs and planning advice instead of the requested single argument |
| Paper boat | Describes a bridge repeatedly; omits the child, boat problem and ending |
| Grandmother follow-up | Repeats questions instead of writing the requested first-person paragraph |

Both exact reading/correction checks failed. The reading answer echoed the
passage instead of returning only the container. The correction answer gave
an unrelated stock capability statement.

## Release criteria

Compare actual text for relevance, requested details, connected sentences,
repetition and paragraph format. Reject a candidate that merely becomes longer,
copies prompt instructions, or improves loss without clearer writing. Keep
the current website model until review is complete. Preserve all parent weights.

## Candidate review

The 400 updates completed. Validation on the fixed 64-example sample moved
from 1.8346 to 1.8174 at the best checkpoint (step 250); final loss was 1.8198.
This did not produce a reliable improvement in the writing checks.

| Prompt | Best checkpoint | Final checkpoint |
| --- | --- | --- |
| Cats | Repeats loyalty claims; no game | Repeats agitation claims; no game |
| Library | Describes a generic quiet space | Contradictory quiet/noise claims and "scent of bullet points" |
| Repair notes | Copies the prompt into its reply | Repetitive planning advice, missing supplied facts |
| Garden | Repeats planting seeds | Repeats planting seeds; missing the argument |
| Boat | Repeated river/bridge sentences | Contradictory hot/cold river details |
| Grandmother | Repeated advice about the sewing box | Repeats sewing-box/sleeve claims; wrong perspective |

Both exact reading/correction checks remained 0/2 for baseline, best and final.
Decision: **withhold both newly trained candidates**.

A paired JavaScript check of the current deployed int8 Big Kani v2 weights
also compared default greedy decoding with repetition penalty 1.1 and frequency
penalty 0.05. This changed wording but did not reliably follow the writing
instructions. It introduced off-topic claims and failed the rewrite request.
An additional bakery prompt produced invented claims about a training club.
Decision: **do not ship the decoding change**. Raw outputs: `decoding.json`.

The retained step-4250 checkpoint from the earlier paragraph run was also
tested on the same eight prompts. It repeated friendship claims for cats,
repeated generic library sentences, returned numbered lists for the rewrite
and garden requests, and invented bridge/boat-club claims for the story.
The follow-up degenerated into repeated bullet characters. Exact checks were
again 0/2. Decision: **withhold this checkpoint too**. Raw outputs:
`earlier-paragraph.json`.

No model weights, decoding settings, website bundle or default CLI build were
changed by this experiment. The data builder and runner improvements remain
available for future, separately evaluated experiments. Eight focused software
tests passed (six tokenizer/supervisor checks and two runner regression checks);
those tests do not establish better writing.
