"""Layer 3 — GPT-4o Oracle.

Responsibilities
----------------
- Handle low-confidence edge cases escalated by Layer 2.
- Build few-shot prompt from 18 seed images (3 per class).
- Call Azure OpenAI gpt-4o with structured JSON output enforced via Pydantic.
- Log prediction and add image to retraining queue for Layer 2.
"""
