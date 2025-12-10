import pytest
from pipeline import detect_intent

# Test cases: (input_text, expected_intent)
test_cases = [
    ("Elon Musk founded SpaceX in 2002.", "fact_claim"),
    ("NASA landed the Perseverance rover on Mars in February 2021.", "fact_claim"),
    ("According to WHO, 'vaccines cause autism'.", "fact_claim"),
    ("Global renewable energy investment reached $1 trillion in 2023.", "fact_claim"),
    ("Did WHO declare COVID-19 a pandemic in March 2020?", "fact_question"),
    ("Tesla launched the Model S Plaid in 2021 and reported 50% revenue growth last year.", "multi_claim"),
    ("I believe electric cars will dominate by 2030.", "opinion"),
    ("Ukraine recaptured Bakhmut in July 2025.", "fact_claim"),
    ("Blue banana sunlight galaxy banana.", "nonsense"),
    ("https://www.bbc.com/news/world-europe-66298489", "fact_claim"),
    ("To reset your iPhone, press and hold the power and volume buttons until the Apple logo appears.", "instructional"),
    ("https://cnn.com/article123 https://nytimes.com/article456", "fact_claim"),
    ("Water boils at 100°C.", "fact_claim"),
    ("🤯🔥🚀🤡💩", "nonsense"),
    ("Because Musk said in 2023", "nonsense"),
    ("OpenAI released GPT-5 in 2025, and I think it's the best model ever built.", "multi_claim"),
    ("In 2025, global emissions dropped by 3%, attributed to widespread adoption of EVs, renewable grid upgrades in the EU, and severe industrial slowdowns in China.", "news_paragraph"),
    ("Apple launched Vision Pro in 2023, acquired a health startup, and hit a $3 trillion market cap all in one year.", "multi_claim"),
    ("As stated by the UN (https://un.org), 'climate change is a myth.'", "fact_claim"),
    ("WHO declared COVID-19 a pandemic in April 2020, not March 2020 like they said.", "fact_claim"),
    ("You won’t believe what NASA discovered on Mars!", "fact_claim"),
    ("What in the fucking world is this?", "nonsense"),
    ("Can jokowi be the next president of Indonesia?", "fact_question"),
    ("Who is prabowo subianto?", "fact_question"),
    ("Nigga balls itch, is it correct?", "nonsense")
]

@pytest.mark.parametrize("input_text, expected_intent", test_cases)
def test_detect_intent(input_text, expected_intent, shared_llm):
    actual = detect_intent(input_text, llm=shared_llm)
    assert actual == expected_intent, f"Expected: {expected_intent}, Got: {actual}, Input: {input_text}"
