from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "oil": ["oil", "brent", "crude", "barrel", "opep", "fuel", "diesel", "gasoline", "refining"],
    "mining": ["iron ore", "ore", "mining", "steel", "nickel", "copper", "pellet"],
    "aviation": ["airline", "aviation", "jet fuel", "flights", "passengers", "airfare"],
    "banking": ["interest rate", "credit", "loan", "default", "bank", "banking", "spread"],
    "technology": ["cloud", "ai", "semiconductor", "software", "data center", "chip"],
    "retail": ["consumer", "retail", "e-commerce", "sales", "store", "shopping"],
    "macro": ["inflation", "gdp", "recession", "rate cut", "rate hike", "unemployment"],
    "government": ["government", "regulation", "tax", "tariff", "sanction", "minister", "president"],
    "dividends": ["dividend", "payout", "yield", "buyback", "jcp"],
    "earnings": ["earnings", "profit", "loss", "ebitda", "revenue", "guidance", "results"],
    "geopolitics": ["war", "conflict", "tension", "attack", "middle east", "supply disruption"],
}

POSITIVE_TERMS = [
    "rise", "rises", "gain", "gains", "up", "beats", "strong", "record", "surge", "improves",
    "growth", "higher", "positive", "dividend", "buyback", "profit", "expands", "recovery"
]
NEGATIVE_TERMS = [
    "fall", "falls", "drop", "drops", "down", "misses", "weak", "loss", "plunge", "cuts",
    "decline", "lower", "negative", "investigation", "accident", "lawsuit", "pressure", "risk"
]

SECTOR_MAP = {
    "energy": {
        "positive_topics": {"oil", "dividends", "earnings", "geopolitics"},
        "negative_topics": {"government"},
        "topic_weights": {"oil": 0.18, "dividends": 0.12, "earnings": 0.10, "geopolitics": 0.06, "government": -0.16},
        "phrase_weights": {
            "oil prices rise": 0.22,
            "crude prices rise": 0.22,
            "higher oil": 0.20,
            "dividend": 0.10,
            "government intervention": -0.25,
            "price controls": -0.25,
            "oil prices fall": -0.22,
            "crude prices fall": -0.22,
        },
    },
    "basic materials": {
        "positive_topics": {"mining", "earnings", "macro"},
        "negative_topics": {"government"},
        "topic_weights": {"mining": 0.16, "earnings": 0.10, "macro": 0.04, "government": -0.08},
        "phrase_weights": {
            "iron ore rises": 0.22,
            "china stimulus": 0.18,
            "iron ore falls": -0.22,
        },
    },
    "industrials": {
        "positive_topics": {"macro", "earnings"},
        "negative_topics": {"government"},
        "topic_weights": {"macro": 0.05, "earnings": 0.10, "government": -0.06},
        "phrase_weights": {},
    },
    "financial services": {
        "positive_topics": {"banking", "earnings", "macro"},
        "negative_topics": {"government"},
        "topic_weights": {"banking": 0.14, "earnings": 0.10, "macro": 0.04, "government": -0.05},
        "phrase_weights": {
            "rate cut": -0.04,
            "rate hike": 0.04,
            "lower defaults": 0.12,
            "higher defaults": -0.14,
        },
    },
    "consumer cyclicals": {
        "positive_topics": {"retail", "macro", "earnings"},
        "negative_topics": {"government"},
        "topic_weights": {"retail": 0.10, "macro": 0.05, "earnings": 0.10, "government": -0.04},
        "phrase_weights": {},
    },
    "technology": {
        "positive_topics": {"technology", "earnings"},
        "negative_topics": {"government", "macro"},
        "topic_weights": {"technology": 0.12, "earnings": 0.10, "government": -0.05, "macro": -0.02},
        "phrase_weights": {},
    },
    "airlines": {
        "positive_topics": {"aviation", "earnings", "macro"},
        "negative_topics": {"oil", "geopolitics"},
        "topic_weights": {"aviation": 0.10, "earnings": 0.10, "macro": 0.03, "oil": -0.18, "geopolitics": -0.08},
        "phrase_weights": {
            "oil prices rise": -0.24,
            "crude prices rise": -0.24,
            "oil prices fall": 0.20,
            "crude prices fall": 0.20,
        },
    },
}

DEFAULT_SECTOR_RULES = {
    "positive_topics": {"earnings", "dividends"},
    "negative_topics": {"government"},
    "topic_weights": {"earnings": 0.08, "dividends": 0.06, "government": -0.05},
    "phrase_weights": {},
}


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def infer_sector_key(sector: str | None, industry: str | None, description: str) -> str:
    joined = " ".join([sector or "", industry or "", description or ""]).lower()
    if any(k in joined for k in ["energy", "oil", "gas", "petroleum", "refining"]):
        return "energy"
    if any(k in joined for k in ["mining", "materials", "ore", "steel", "metals"]):
        return "basic materials"
    if any(k in joined for k in ["bank", "financial", "insurance", "credit"]):
        return "financial services"
    if any(k in joined for k in ["airline", "aviation", "air transport"]):
        return "airlines"
    if any(k in joined for k in ["technology", "software", "semiconductor", "it services"]):
        return "technology"
    if any(k in joined for k in ["retail", "e-commerce", "consumer cyclical"]):
        return "consumer cyclicals"
    if any(k in joined for k in ["industrial", "capital goods", "machinery"]):
        return "industrials"
    return "default"


def detect_topics(text: str) -> List[str]:
    t = normalize(text)
    found = []
    for topic, words in TOPIC_KEYWORDS.items():
        if any(w in t for w in words):
            found.append(topic)
    return found


def score_relevance(news_text: str, company_name: str, ticker: str, description: str, sector: str | None, industry: str | None) -> float:
    text = normalize(news_text)
    name = normalize(company_name)
    ticker_n = normalize(ticker).replace(".sa", "")
    desc = normalize(description)

    score = 0.1
    if name and any(tok in text for tok in [name, ticker_n]):
        score += 0.45

    company_terms = [w for w in re.findall(r"[a-zA-Z]{4,}", desc) if w not in {"company", "business", "services"}]
    overlap = sum(1 for term in set(company_terms[:20]) if term in text)
    score += min(overlap * 0.05, 0.25)

    sector_key = infer_sector_key(sector, industry, description)
    sector_rules = SECTOR_MAP.get(sector_key, DEFAULT_SECTOR_RULES)
    topics = set(detect_topics(text))
    relevant_topic_hits = topics.intersection(set(sector_rules["topic_weights"].keys()))
    score += min(len(relevant_topic_hits) * 0.08, 0.20)

    return max(0.0, min(1.0, score))


def base_sentiment(news_text: str) -> float:
    return float(analyzer.polarity_scores(news_text)["compound"])


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    t = normalize(text)
    return sorted({k for k in keywords if k in t})


def sector_adjustment(news_text: str, sector: str | None, industry: str | None, description: str) -> Tuple[float, List[str], List[str], List[str]]:
    text = normalize(news_text)
    topics = detect_topics(text)
    sector_key = infer_sector_key(sector, industry, description)
    rules = SECTOR_MAP.get(sector_key, DEFAULT_SECTOR_RULES)

    adjustment = 0.0
    positive_signals: List[str] = []
    negative_signals: List[str] = []

    for topic in topics:
        weight = rules["topic_weights"].get(topic, 0.0)
        adjustment += weight
        if weight > 0:
            positive_signals.append(f"topic:{topic}")
        elif weight < 0:
            negative_signals.append(f"topic:{topic}")

    for phrase, weight in rules["phrase_weights"].items():
        if phrase in text:
            adjustment += weight
            if weight > 0:
                positive_signals.append(f"phrase:{phrase}")
            elif weight < 0:
                negative_signals.append(f"phrase:{phrase}")

    pos_terms = keyword_hits(text, POSITIVE_TERMS)
    neg_terms = keyword_hits(text, NEGATIVE_TERMS)
    adjustment += min(len(pos_terms) * 0.015, 0.06)
    adjustment -= min(len(neg_terms) * 0.015, 0.06)
    positive_signals.extend([f"word:{w}" for w in pos_terms])
    negative_signals.extend([f"word:{w}" for w in neg_terms])

    return adjustment, topics, sorted(set(positive_signals)), sorted(set(negative_signals))


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def label_from_score(score: float) -> str:
    if score >= 0.15:
        return "positivo"
    if score <= -0.15:
        return "negativo"
    return "neutro"


def confidence_from_scores(base: float, relevance: float, adjustment: float) -> float:
    raw = 0.35 + (abs(base) * 0.25) + (relevance * 0.25) + min(abs(adjustment), 0.4)
    return round(max(0.0, min(1.0, raw)), 3)


def explain(company_name: str, sector_key: str, topics: List[str], final_score: float, pos: List[str], neg: List[str]) -> str:
    direction = "positivo" if final_score > 0.15 else "negativo" if final_score < -0.15 else "neutro"
    topic_text = ", ".join(topics) if topics else "sem tópicos claros"
    signals = pos[:2] + neg[:2]
    signal_text = ", ".join(signals) if signals else "sem sinais fortes"
    return (
        f"Impacto {direction} para {company_name}. O motor identificou contexto setorial '{sector_key}' "
        f"e temas {topic_text}. Principais sinais: {signal_text}."
    )


def analyze_news_for_company(company: Dict, news: Dict) -> Dict:
    title = news.get("title", "")
    summary = news.get("summary", "") or ""
    news_text = f"{title}. {summary}".strip()

    comp_name = company.get("name", "")
    ticker = company.get("ticker", "")
    sector = company.get("sector")
    industry = company.get("industry")
    description = company.get("description", "")

    base = base_sentiment(news_text)
    relevance = score_relevance(news_text, comp_name, ticker, description, sector, industry)
    adjustment, topics, pos, neg = sector_adjustment(news_text, sector, industry, description)

    # Relevance gates the adjustment so unrelated news does not distort the result.
    gated_adjustment = adjustment * (0.35 + 0.65 * relevance)
    final = clamp(base + gated_adjustment)
    sector_key = infer_sector_key(sector, industry, description)

    return {
        "ticker": ticker,
        "company_name": comp_name,
        "base_sentiment_score": round(base, 4),
        "relevance_score": round(relevance, 4),
        "adjustment_score": round(gated_adjustment, 4),
        "final_sentiment_score": round(final, 4),
        "sentiment_label": label_from_score(final),
        "confidence": confidence_from_scores(base, relevance, gated_adjustment),
        "detected_topics": topics,
        "positive_signals": pos,
        "negative_signals": neg,
        "explanation": explain(comp_name, sector_key, topics, final, pos, neg),
    }
