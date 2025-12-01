# services/classifier_intent/prompt.py

def build_detect_intent_system() -> str:
    return (
        "You are an intent classifier. "
        "Return exactly one of: fact_claim, fact_question, opinion, nonsense, "
        "news_paragraph, multi_claim, instructional. "
        "Return only the label. No punctuation. No explanation."
    )

def build_detect_intent_prompt(text: str) -> str:
    return (
        "Classify this text into exactly one category.\n\n"
        "Categories:\n"
        "- fact_claim: A single factual statement to verify (e.g., 'Elon Musk founded SpaceX in 2002')\n"
        "- fact_question: A question asking about facts (e.g., 'Did NASA land on Mars?')\n"
        "- opinion: A personal belief / subjective view (e.g., 'I think cats are better than dogs')\n"
        "- nonsense: Meaningless, gibberish, emojis spam, or incoherent fragment\n"
        "- news_paragraph: A paragraph containing multiple factual statements (longer narrative)\n"
        "- multi_claim: A single sentence containing multiple distinct factual claims (linked by 'and', commas, semicolons)\n"
        "- instructional: A how-to or procedural instruction (e.g., 'To reset your iPhone, press ...')\n\n"
        f"Text: \"{text}\"\n\n"
        "Answer with only one category name (exact label)."
    )
