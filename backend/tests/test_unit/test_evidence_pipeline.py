import pytest
from pipeline.factcheck import evidence as ev_mod

# -----------------------------
# Fixtures
# -----------------------------

class DummyLLM:
    def __init__(self, response_map=None, default="No relevant evidence found."):
        self.response_map = response_map or {}
        self.default = default
        self.calls = []

    def generate_response(self, prompt, max_tokens=128):
        self.calls.append(prompt)
        for key, value in self.response_map.items():
            if key in prompt:
                return value
        return self.default

    def clear_cache(self):
        pass


@pytest.fixture
def dummy_llm_exact():
    return DummyLLM(response_map={"Select up to 2 sentences": "First sentence evidence"})


@pytest.fixture
def dummy_llm_hallucinating():
    return DummyLLM(response_map={"Select up to 2 sentences": "Totally made up sentence"})


@pytest.fixture
def dummy_llm_none():
    return DummyLLM(default="No relevant evidence found.")


# Common minimal search response helper
def make_search_resp(urls_with_titles):
    items = []
    for title, url in urls_with_titles:
        items.append({"title": title, "link": url, "snippet": f"Snippet for {title}"})
    return {"items": items}


# -----------------------------
# Monkeypatch helpers
# -----------------------------

@pytest.fixture
def patch_fetch_article_text(monkeypatch):
    def _apply(mapping):
        def fake_fetch(url):
            return mapping.get(url, "")
        monkeypatch.setattr(ev_mod, "fetch_article_text", fake_fetch)
    return _apply

@pytest.fixture
def patch_best_sentences(monkeypatch):
    def _apply(return_map):
        def fake_best(text, claim, k):
            return return_map.get(text, [])[:k]
        monkeypatch.setattr(ev_mod, "best_sentences", fake_best)
    return _apply

@pytest.fixture
def patch_news_api(monkeypatch):
    def _apply(items):
        def fake_news(claim, max_results=2):
            return items[:max_results]
        monkeypatch.setattr(ev_mod, "fetch_newsapi_articles", fake_news)
    return _apply

@pytest.fixture
def patch_attribution(monkeypatch):
    def _apply(value):
        monkeypatch.setattr(ev_mod, "attribution_tail", lambda claim: value)
    return _apply


# -----------------------------
# Tests
# -----------------------------

def test_collect_search_items_merges(monkeypatch, patch_news_api):
    patch_news_api([{"title": "NewsA", "link": "http://n1", "snippet": "n1"}])
    sr = make_search_resp([("G1", "http://g1"), ("G2", "http://g2"), ("G3", "http://g3")])
    items = ev_mod.collect_search_items(sr, "Some claim", max_google=2, max_news=1)
    assert len(items) == 3
    assert items[0]["source"] == "Google"
    assert items[-1]["source"] == "NewsAPI"


def test_fetch_article_candidates_literal_hit(patch_fetch_article_text, patch_best_sentences, patch_attribution):
    patch_attribution("Mars landing")
    fake_text = "A long article about a Mars landing milestone. Another Mars landing test sentence."
    patch_fetch_article_text({"http://g1": fake_text})
    patch_best_sentences({fake_text: [
        "A long article about a Mars landing milestone.",
        "Another Mars landing test sentence.",
        "Irrelevant filler."
    ]})

    items = [{
        "title": "G1",
        "url": "http://g1",
        "snippet": "s",
        "source": "Google"
    }]
    cands, urls = ev_mod.fetch_article_candidates(items, "NASA landed humans on Mars in 2023", sents_per_article=2, literal_phrase="Mars landing")
    assert urls == ["http://g1"]
    # Ensure literal flagged
    assert any(c.get("literal") for c in cands)


def test_rank_source_quotes_selects_top(patch_fetch_article_text, patch_best_sentences):
    fake_text = "One. Two. Three."
    patch_fetch_article_text({"http://a": fake_text})
    patch_best_sentences({fake_text: [
        "Sentence about GPT-5 release rumor.",
        "Second sentence with less relevance.",
        "Another unrelated line."
    ]})

    items = [{
        "title": "T1",
        "url": "http://a",
        "snippet": "s",
        "source": "Google"
    }]
    cands, _ = ev_mod.fetch_article_candidates(items, "GPT-5 release date", sents_per_article=3, literal_phrase=None)
    quotes = ev_mod.rank_source_quotes("GPT-5 release date", cands, top_k=2)
    assert len(quotes) == 2
    assert all("quote" in q for q in quotes)


def test_select_evidence_llm_exact(dummy_llm_exact):
    candidate_sents = [
        {"text": "First sentence evidence", "source": "S1", "url": "u1", "literal": False},
        {"text": "Second sentence filler", "source": "S2", "url": "u2", "literal": False},
    ]
    ev = ev_mod.select_evidence("Some claim", candidate_sents, dummy_llm_exact, 40)
    assert ev == "First sentence evidence"


def test_select_evidence_llm_hallucination_fallback(dummy_llm_hallucinating):
    candidate_sents = [
        {"text": "Legitimate sentence A", "source": "S1", "url": "u1", "literal": False},
        {"text": "Legitimate sentence B", "source": "S2", "url": "u2", "literal": False},
    ]
    ev = ev_mod.select_evidence("Some claim", candidate_sents, dummy_llm_hallucinating, 40)
    assert ev in ["Legitimate sentence A", "Legitimate sentence B"]


def test_select_evidence_literal_priority(dummy_llm_none):
    candidate_sents = [
        {"text": "Literal match exact phrase here", "source": "S1", "url": "u1", "literal": True},
        {"text": "Some other sentence", "source": "S2", "url": "u2", "literal": False},
    ]
    ev = ev_mod.select_evidence("Claim", candidate_sents, dummy_llm_none, 40)
    assert ev.startswith("Literal match")


def test_select_evidence_no_llm_fallback():
    candidate_sents = [
        {"text": "Alpha relevance", "source": "S1", "url": "u1", "literal": False},
        {"text": "Beta something else", "source": "S2", "url": "u2", "literal": False},
    ]
    ev = ev_mod.select_evidence("Alpha thing", candidate_sents, llm=None, max_words=8)
    assert isinstance(ev, str)
    assert len(ev.split()) <= 8


def test_build_evidence_end_to_end_minimal(monkeypatch, patch_news_api, patch_fetch_article_text, patch_best_sentences, patch_attribution, dummy_llm_exact):
    # Fake search
    search_resp = {"items": [
        {"title": "Doc1", "link": "http://doc1", "snippet": "snip1"},
    ]}
    # Fake NewsAPI
    patch_news_api([
        {"title": "News1", "link": "http://n1", "snippet": "nsnip"}
    ])
    # Fake article texts
    patch_fetch_article_text({
        "http://doc1": "GPT-5 speculation grows. OpenAI executives deny release.",
        "http://n1": "Analysts discuss GPT-5 timeline uncertainty."
    })
    # Each text returns sentences
    patch_best_sentences({
        "GPT-5 speculation grows. OpenAI executives deny release.": [
            "OpenAI executives deny release.",
            "GPT-5 speculation grows."
        ],
        "Analysts discuss GPT-5 timeline uncertainty.": [
            "Analysts discuss GPT-5 timeline uncertainty."
        ]
    })
    patch_attribution(None)

    evidence, urls, quotes = ev_mod.build_evidence(
        search_resp=search_resp,
        claim="OpenAI released GPT-5 in January 2025",
        llm=dummy_llm_exact,
        sents_per_article=2
    )

    assert evidence
    assert len(urls) == 2
    assert 0 < len(quotes) <= 3


def test_build_evidence_no_candidates(monkeypatch, patch_news_api, patch_fetch_article_text, patch_best_sentences, patch_attribution):
    search_resp = {"items": [
        {"title": "Doc1", "link": "http://doc1", "snippet": "snip1"},
    ]}
    patch_news_api([])
    patch_fetch_article_text({"http://doc1": ""})  # no content
    patch_best_sentences({})
    patch_attribution(None)

    evidence, urls, quotes = ev_mod.build_evidence(
        search_resp=search_resp,
        claim="Some claim",
        llm=None,
        sents_per_article=2
    )

    assert evidence == ""
    assert urls == []
    assert quotes == []