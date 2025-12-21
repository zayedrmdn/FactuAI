# Source Filtering (The Gatekeeper)

Strict filtering of search results to block social media and low-quality sources.

---

## Purpose

**Problem:** Social media platforms are vectors for misinformation, echo chambers, and unverified claims.

**Solution:** Block all social media sources at the search provider level (Tavily) before results enter the system.

---

## The Social Media Blocklist

### Blocked Domains (20 total)

```python
SOCIAL_MEDIA_DOMAINS = [
    "facebook.com",
    "fb.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "instagram.com",
    "youtube.com",
    "linkedin.com",
    "pinterest.com",
    "snapchat.com",
    "medium.com",
    "substack.com",
    "quora.com",
    "tumblr.com",
    "vk.com",
    "weibo.com",
    "telegram.org",
    "discord.com",
    "vimeo.com",
    "wikipedia.org"  # User-generated, often disputed
]
```

**Note:** Wikipedia is excluded because it's user-generated and often contains disputes/edit wars that shouldn't be used as primary sources for fact-checking.

---

## Implementation

### Tavily Integration

**Location:** `backend/app/features/search/providers/tavily.py`

```python
from app.core.constants import SOCIAL_MEDIA_DOMAINS

async def search(self, query: str):
    response = await self.client.search(
        query=query,
        exclude_domains=SOCIAL_MEDIA_DOMAINS,  # ← The gatekeeper
        include_answer=True,
        include_raw_content=False  # Disabled to prevent HTML artifacts
    )
    return response
```

**When:** Filtering happens at the Tavily API level (before results are returned)

**Guarantee:** Social media URLs **never** enter the FactuAI system

---

## Rationale by Domain Type

### Social Networks (Facebook, Twitter/X, Instagram, TikTok)
- High misinformation spread
- Echo chambers and filter bubbles
- Unverified user-generated content
- Emotionally charged sharing

### Discussion Forums (Reddit, Quora)
- Opinion-based, not fact-based
- Upvote systems favor popular, not accurate
- Anonymous users, no accountability

### User-Generated Platforms (Medium, Substack, YouTube)
- No editorial oversight
- Anyone can publish anything
- Variable quality

### Wikipedia
- Constantly changing
- Edit wars on controversial topics
- Better as a secondary source, not primary

---

## What We DO Accept

**Prioritized Sources:**
- ✅ Fact-check sites (Snopes, PolitiFact, FactCheck.org)
- ✅ Primary sources (NASA, CDC, WHO, government agencies)
- ✅ Peer-reviewed journals (Nature, Science, JAMA)
- ✅ Reputable news outlets (AP, Reuters, BBC)
- ✅ Academic institutions (.edu domains)

---

## Edge Cases

### 1. YouTube for Official Channels

**Problem:** YouTube is blocked, but official channels (NASA, WHO) post there

**Solution:** Official transcripts/websites are preferred; if YouTube is the only source, it doesn't make it through

**Future:** Whitelist specific verified channels

### 2. Wikipedia for Quick Reference

**Problem:** Wikipedia is useful for background context

**Workaround:** Users can manually reference Wikipedia; system won't cite it as evidence

### 3. Legitimate Reddit AMAs

**Problem:** Expert AMAs (Ask Me Anything) on Reddit can be valuable

**Current:** Blocked (Reddit is not allowed)

**Future:** Consider whitelist for r/science verified AMAs

---

## Configuration

### Adding/Removing Domains

**File:** `backend/app/core/constants.py`

```python
SOCIAL_MEDIA_DOMAINS = [
    # Add new domains here
    "newplatform.com",
]
```

**After changes:** Restart backend

---

## Monitoring

### Check Blocked Sources

```sql
-- This query should return 0 rows if filtering works
SELECT url, domain 
FROM sources 
WHERE domain IN ('facebook.com', 'twitter.com', 'reddit.com');
```

### Source Domain Distribution

```sql
-- Top 10 source domains in database
SELECT domain, COUNT(*) as count
FROM sources
GROUP BY domain
ORDER BY count DESC
LIMIT 10;
```

**Expected:** Fact-check sites, news outlets, official sources

---

## Testing

```python
# backend/tests/test_source_filtering.py
async def test_no_social_media_sources():
    results = await tavily_provider.search("vaccines cause autism")
    
    for result in results:
        domain = extract_domain(result.url)
        assert domain not in SOCIAL_MEDIA_DOMAINS
```

---

## Impact on Results

### Before Filtering (Hypothetical)

**Search:** "Do vaccines cause autism?"

**Results:**
1. ❌ Facebook post claiming vaccines are dangerous
2. ❌ Reddit thread with anecdotal stories
3. ✅ CDC study showing no link
4. ❌ YouTube video spreading misinformation
5. ✅ Snopes fact-check debunking the claim

**Problem:** 60% low-quality sources

### After Filtering (Actual)

**Results:**
1. ✅ CDC study showing no link
2. ✅ Snopes fact-check debunking the claim
3. ✅ WHO statement on vaccine safety
4. ✅ Peer-reviewed study in The Lancet
5. ✅ PolitiFact rating: "Pants on Fire"

**Quality:** 100% credible sources

---

## Constitution Rule

From [constitution.md](../01-rules/constitution.md):

> ### 10) Strict Source Filtering (The Gatekeeper)
>
> - All external search results must be filtered against a **Social Media Blocklist** before ingestion.
> - Blocked domains include: Facebook, TikTok, Twitter/X, Reddit, Instagram, YouTube, Medium, Wikipedia.
> - Implementation: `TavilySearchProvider` uses `exclude_domains` with `SOCIAL_MEDIA_DOMAINS` constant.
> - **Rationale**: Social media is a vector for misinformation; we prioritize primary sources and fact-check sites.

---

## Code Pointers

- Blocklist constant: `backend/app/core/constants.py`
- Tavily implementation: `backend/app/features/search/providers/tavily.py`
- Constitution rule: [docs/01-rules/constitution.md](../01-rules/constitution.md)

---

See also:
- [Phase 2: Parallel Search](../04-pipeline/03-search.md)
- [Constitution](../01-rules/constitution.md)
- [Phase 1: Strategist](../04-pipeline/02-strategist.md) - Query generation
