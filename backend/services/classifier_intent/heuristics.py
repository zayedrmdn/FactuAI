# services/classifier_intent/heuristics.py

import re
from typing import Optional

OPINION_MARKERS = {
    "i think", "i believe", "i feel", "personally", "in my opinion", "imo"
}
INSTRUCTION_START = (
    "to ", "first ", "step ", "press ", "hold ", "click ", "open ", "install "
)
NONSENSE_REGEX = re.compile(
    r"^(?:[\W_]*|[\W_\d]*)(?:[^\w\s]|[\u2600-\u27BF]|[\U0001F300-\U0001FAFF]){2,}.*$"
)

def _looks_multi_claim(text: str) -> bool:
    clauses = re.split(r",|;| and ", text)
    verbs = (
        "launched", "acquired", "reported", "announced", "hit",
        "reached", "built", "released", "developed"
    )
    hits = sum(1 for c in clauses if any(v in c.lower() for v in verbs))
    return hits >= 2

def _fast_heuristics(text: str) -> Optional[str]:
    low = text.lower()
    words = text.split()
    wc = len(words)

    # 1) Pure gibberish / emoji spam
    if NONSENSE_REGEX.match(text):
        return "nonsense"

    # 2) Very short noise or subordinate fragment
    if ((wc <= 5 and
         not any(ch.isdigit() for ch in text) and
         not re.search(r"\b\w+(?:ed|ing)\b", low))
        or (low.startswith(("because ", "since ", "if ", "when ")) and wc <= 10)):
        return "nonsense"

    # 3) Multi-claim: two+ factual verbs
    if _looks_multi_claim(text):
        return "multi_claim"

    # 4) News paragraph: long or multi-sentence
    if wc > 20 or text.count(". ") >= 2:
        return "news_paragraph"

    # 5) Opinion markers
    if any(marker in low for marker in OPINION_MARKERS):
        return "opinion"

    # 6) Instructional patterns
    if low.startswith(INSTRUCTION_START) or low.startswith("how to "):
        return "instructional"

    return None
