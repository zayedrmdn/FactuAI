# services/classifier_intent/parser.py

import re
from typing import Optional

INTENT_CLASSES = [
    "fact_claim",
    "fact_question",
    "opinion",
    "news_paragraph",
    "nonsense",
    "multi_claim",
    "instructional"
]

def _parse_llm_label(raw: str) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip().strip('"').strip("'").lower()
    if cleaned in INTENT_CLASSES:
        return cleaned
    # try JSON-like extraction
    m = re.search(r'"?intent"?\s*:\s*"?(?P<label>[a-z_]+)"?', cleaned)
    if m and m.group("label") in INTENT_CLASSES:
        return m.group("label")
    # find any token matching
    for token in re.findall(r"[a-z_]+", cleaned):
        if token in INTENT_CLASSES:
            return token
    return None
