# Instruction data sources

Databricks Dolly 15k contributors — CC BY-SA 3.0: https://huggingface.co/datasets/databricks/databricks-dolly-15k

OpenAssistant OASST2 contributors — Apache-2.0: https://huggingface.co/datasets/OpenAssistant/oasst2

This filtered derivative preserves source identifiers, removes exact repeated prompts/replies and overly long examples, and formats retained conversations for Kani. Dolly-derived records remain subject to CC BY-SA 3.0. Source hashes and selection rules are in manifest.json and data/make_advanced_instructions.py.

Paragraph selection and prompt formatting: data/make_paragraphs.py. Original source identifiers and licenses are retained.
