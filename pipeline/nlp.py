"""NLP analysis: keywords, NER, summarization, sentiment, TF-IDF, classification.

Public functions use module-level cached models by default for convenience.
Pass explicit model instances when you need testability or custom config.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import DocRecord

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model caches (lazy-loaded singletons)
# ---------------------------------------------------------------------------

_nlp = None
_rake = None
_vader = None


def _get_spacy():
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.warning(
                "spaCy model en_core_web_sm not found; using blank English pipeline"
            )
            _nlp = spacy.blank("en")
            if "sentencizer" not in _nlp.pipe_names:
                _nlp.add_pipe("sentencizer")
    return _nlp


def _get_rake():
    global _rake
    if _rake is None:
        from rake_nltk import Rake

        _rake = Rake()
    return _rake


def _get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader = SentimentIntensityAnalyzer()
    return _vader


def init_nlp() -> tuple:
    """Load all NLP models once. Returns (spacy_nlp, rake, vader)."""
    import nltk

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    return _get_spacy(), _get_rake(), _get_vader()


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def extract_keywords_rake(text: str, top_n: int = 15) -> list[str]:
    """Extract keywords using RAKE algorithm."""
    rake = _get_rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:top_n]


# Alias
extract_keywords = extract_keywords_rake


# ---------------------------------------------------------------------------
# Named Entity Recognition
# ---------------------------------------------------------------------------


def extract_entities(text: str, max_chars: int = 100_000) -> dict[str, list[str]]:
    """Extract named entities grouped by label using spaCy."""
    nlp = _get_spacy()
    doc = nlp(text[:max_chars])
    entities: dict[str, set[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, set()).add(ent.text.strip())
    return {k: sorted(v)[:20] for k, v in entities.items()}


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def extractive_summary(text: str, num_sentences: int = 5) -> str:
    """Fast extractive summary -- first N non-trivial sentences."""
    nlp = _get_spacy()
    doc = nlp(text[:50_000])
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 40]
    return " ".join(sentences[:num_sentences])


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------


def analyze_sentiment(text: str, analyzer: Any = None) -> dict[str, Any]:
    """Sentiment analysis.

    When *analyzer* is provided (a HuggingFace transformers pipeline),
    uses it for backward compatibility with existing tests.
    Otherwise defaults to VADER (lighter, no GPU required).
    """
    if analyzer is not None:
        # HuggingFace transformers pipeline path
        result = analyzer(text[:2000])
        return {
            "label": result[0]["label"],
            "score": round(result[0]["score"], 4),
        }

    # Default: VADER sentiment
    vader = _get_vader()
    scores = vader.polarity_scores(text[:5000])
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return {
        "label": label,
        "score": round(abs(compound), 4),
        "compound": round(compound, 4),
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# TF-IDF topic extraction
# ---------------------------------------------------------------------------


def extract_tfidf_topics(texts: list[str], top_n: int = 10) -> list[list[str]]:
    """Extract top TF-IDF terms per document for topic signals."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not texts:
        return []
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", max_df=0.85, min_df=1
    )
    matrix = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    topics: list[list[str]] = []
    for i in range(matrix.shape[0]):
        row = matrix[i].toarray().flatten()
        top_idx = row.argsort()[-top_n:][::-1]
        topics.append([features[j] for j in top_idx if row[j] > 0])
    return topics


# Alias
compute_tfidf_topics = extract_tfidf_topics


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------


def classify_document_type(text: str) -> str:
    """Heuristic document type classification based on content signals."""
    text_lower = text[:10_000].lower()
    patterns = {
        "lease": ["lease", "landlord", "tenant", "rent", "premises"],
        "contract": [
            "agreement",
            "parties",
            "hereby",
            "whereas",
            "terms and conditions",
        ],
        "invoice": [
            "invoice",
            "amount due",
            "bill to",
            "payment terms",
            "total due",
        ],
        "legal_notice": [
            "notice",
            "hereby notified",
            "pursuant to",
            "demand",
        ],
        "tax_document": [
            "tax",
            "assessment",
            "property tax",
            "assessed value",
            "taxable",
        ],
        "title_report": [
            "title",
            "escrow",
            "deed",
            "recording",
            "conveyance",
        ],
        "amendment": [
            "amendment",
            "first amendment",
            "second amendment",
            "modify",
        ],
        "letter": ["dear", "sincerely", "regards", "attention"],
        "report": ["report", "findings", "analysis", "recommendation"],
        "insurance": [
            "insurance",
            "policy",
            "premium",
            "coverage",
            "claim",
        ],
    }
    scores = {
        doc_type: sum(1 for t in terms if t in text_lower)
        for doc_type, terms in patterns.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


# ---------------------------------------------------------------------------
# Full NLP orchestration
# ---------------------------------------------------------------------------


def run_nlp_analysis(
    records: list[DocRecord],
    sentiment_analyzer: Any = None,
) -> dict[str, dict[str, Any]]:
    """Run full NLP pipeline on successful records.

    Returns ``{filename: analysis_dict}``.
    """
    successful = [r for r in records if r.status == "success" and r.markdown]
    all_texts = [r.markdown for r in successful]
    all_tfidf = extract_tfidf_topics(all_texts)

    results: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(successful):
        text = record.markdown
        entities = extract_entities(text)
        results[record.filename] = {
            "keywords_rake": extract_keywords_rake(text),
            "named_entities": entities,
            "summary": extractive_summary(text),
            "sentiment": analyze_sentiment(text, analyzer=sentiment_analyzer),
            "tfidf_topics": all_tfidf[i] if i < len(all_tfidf) else [],
            "document_type": classify_document_type(text),
            "word_count": len(text.split()),
            "char_count": len(text),
            "people": entities.get("PERSON", []),
            "organizations": entities.get("ORG", []),
            "dates": entities.get("DATE", []),
            "amounts": entities.get("MONEY", []),
        }
    return results
