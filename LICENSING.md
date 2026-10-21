# Reusing Kani Seal 3

Copyright 2026 the Kelus Model B1 contributors.

Kani Seal 3, formerly Kelus Model B1, retains the same attribution and
license. Its original project materials are licensed under the
**Apache License, Version 2.0** (`Apache-2.0`). The complete, unmodified
[LICENSE](LICENSE) is authoritative; this guide explains its application
and adds no license conditions. [NOTICE](NOTICE) contains the project credit.

## What you may do

- Use B1 for personal, research, educational, or commercial projects.
- Fork the repository, modify its code, and distribute your version.
- Build another AI using B1's implementation and tokenizer.
- Train new models, and fine-tune or adapt B1 weights released under this license.
- Sell products incorporating B1. Apache-2.0 does not require your own
  modifications to be open source.

## Credit when redistributing B1 material

When distributing B1 or a derivative covered by the license, follow section 4:

1. Give recipients a copy of the Apache-2.0 license.
2. Put prominent notices in modified files stating that you changed them.
3. Preserve applicable copyright, patent, trademark, and attribution notices
   in distributed derivative source code.
4. Preserve the applicable attribution from `NOTICE` in a readable form:
   an accompanying notice file, distributed source or documentation, or an
   appropriate third-party-notices display, as permitted by section 4(d).

For a GitHub fork containing B1, keeping `LICENSE` and `NOTICE` at the root
and documenting changes is a practical starting point. For a model release
containing covered B1 weights, include those files with the weights and
identify the B1 base model and your changes in the model card.

Suggested additional README or model-card credit:

> Based on Kelus Model B1 by the Kelus Model B1 contributors, licensed under
> Apache-2.0. This version contains modifications by its own authors.

This suggested wording supplements the required notices; it does not replace
the license copy or the applicable `NOTICE` content.

## Attribution, ownership, and project identity

Removing required attribution or redistributing covered B1 material without
the required license and notices violates the redistribution conditions.
You may claim authorship of your own changes while preserving the original
B1 credits. The license grants reuse rights, not ownership of other authors'
work.

Forking, copying, renaming, and selling a compliant derivative are allowed.
There is no additional, undefined "no theft" restriction on those permitted
activities. Section 6 grants no general right to use the project's names or
trademarks, beyond customary origin descriptions and reproducing notices;
it does not authorize presenting a fork as an official Kelus release.

Apache-2.0's redistribution duties do not impose a credit banner on every
hosted service or every independently trained model. Using the trainer alone
does not automatically make its outputs a derivative of the trainer.
Whether particular weights or outputs contain protected third-party material
depends on those artifacts; this guide does not settle that question.

## Scope and third-party materials

The project license applies to the contributors' rights in B1's original
source code, documentation, original curriculum data, original tokenizer
assets, and model weights distributed by the project under this license,
unless a file or release carries a separate notice. It grants only rights
the contributors can grant. It does not relicense third-party material or
transfer third-party ownership.

In particular:

- **Firebase SDK:** `site/vendor/firebase.js` contains third-party code and
  retained license comments, including Google copyright notices and
  Apache-2.0 terms. Preserve those notices when redistributing the bundle.
- **Dependencies:** NumPy and other separately installed dependencies retain
  their own licenses. Include their required notices if you bundle them.
- **Foundation corpus:** downloaded CPython and MDN material, extracts, and
  tokenized copies retain their applicable upstream terms. Source URLs and
  license references are recorded in `data/foundation/sources.json`,
  `documents.jsonl`, and `licenses/` when the corpus is prepared. See
  [FOUNDATION_TRAINING.md](FOUNDATION_TRAINING.md) for provenance. The root
  Apache license does not replace the Python license stack or MDN's content
  and code licenses.
- **External tokenizer:** the standalone root `tokenizer.json`, if present,
  is the Google Gemma tokenizer supplied for inspection. It is not B1's own
  tokenizer, is not used by the current training run, and is excluded from
  this project's grant. Its original terms apply independently.

Model outputs are not automatically assigned this repository's license.
This license also does not establish that a model release satisfies every
requirement of an open-source-AI definition.

## References

- [Apache-2.0, including redistribution and trademark terms](https://www.apache.org/licenses/LICENSE-2.0)
- [Apache's guidance on applying the license](https://www.apache.org/foundation/license-faq.html#Apply)
- [Open Source Definition: redistribution and derived works](https://opensource.org/osd)
