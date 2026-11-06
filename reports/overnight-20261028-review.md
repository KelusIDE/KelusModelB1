# Overnight candidate review — 2026-10-28

Release decision: **withhold the topic candidate**. Baseline, best and final
checkpoints each scored 1/9 on the narrow automatic suite. A lower training
loss did not establish improved answers. No candidate weights were deployed.

| Manual case | Observed candidate behavior | Assessment |
| --- | --- | --- |
| World hunger | Still returns a generic coding refusal | Fails relevance |
| Origin of Species author | Invents Frank Herbert where baseline requested search | Regression |
| Change topic to cats | Repetitive, odd claims including sleeping in a cat | Fails coherence |
| Adopting a puppy | Echoes excitement without addressing adoption | Fails request |
| Feeling excluded | Repeats a generic loneliness response | No demonstrated gain |
| Parks paragraph | Coherent paragraph, omits the requested example | Partial |
| Rewrite request | Unrelated apology and question | Fails instruction |

Evidence: `training-runs/overnight-20261028/results.json` and the topic
`baseline.json`, `best.json`, and `final.json`. Manual judgments above are
qualitative; they are not a broad benchmark score.

The development pass separately adds bounded, explicitly labeled calculator
phrasing and connects the CLI's expert to existing Cide 2 weights. Passing
software checks for those changes does not establish neural-model gains.
