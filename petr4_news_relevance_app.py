#!/usr/bin/env python3
"""
Protótipo local para buscar notícias da PETR4 no Yahoo Finance, extrair temas,
calcular relevância para o ativo e classificar impacto provável.

Uso:
    python petr4_news_relevance_app.py
    python petr4_news_relevance_app.py --ticker PETR4.SA --count 20
    python petr4_news_relevance_app.py --query "Petrobras" --output resultados.csv

Dependências:
    pip install yfinance pandas

Observações:
- Usa Yahoo Finance via yfinance para buscar notícias do ticker.
- A API de notícias do Yahoo/yfinance pode ser intermitente; por isso o código
  tenta múltiplas rotas e aplica filtragem local adicional.
- A classificação de relevância/impacto aqui é heurística e pensada como V1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf


# ==========================
# Configuração do ativo
# ==========================
ACOES: Dict[str, Dict[str, Any]] = {
    "PETR4.SA": {
        "ticker": "PETR4.SA",
        "nome": "Petrobras",
        "empresa_aliases": [
            "petrobras", "petróleo brasileiro", "petroleo brasileiro", "pbr", "petr4"
        ],
        "temas": ["petroleo", "geopolitica", "governo", "cambio", "dividendos", "resultado"],
        "entidades": [
            "petrobras", "brent", "opep", "diesel", "gasolina", "pre-sal", "pré-sal",
            "refino", "combustível", "combustiveis", "produção", "producao"
        ],
        "descricao_semantica": (
            "Petrobras é uma petroleira brasileira sensível a petróleo Brent, OPEP, "
            "combustíveis, decisões do governo, dividendos, produção, refino, câmbio "
            "e geopolítica energética."
        ),
    }
}


TEMAS: Dict[str, List[str]] = {
    "petroleo": [
        "petróleo", "petroleo", "brent", "barril", "opep", "oferta", "produção", "producao",
        "refino", "combustível", "combustiveis", "diesel", "gasolina", "energia", "pré-sal", "pre-sal"
    ],
    "geopolitica": [
        "guerra", "conflito", "sanção", "sancao", "ataque", "tensão", "tensao",
        "bloqueio", "oriente médio", "oriente medio", "russia", "ucrânia", "ucrania",
        "israel", "irã", "ira", "estreito", "rotas marítimas", "rotas maritimas"
    ],
    "governo": [
        "governo", "presidente", "ministério", "ministerio", "regulação", "regulacao",
        "estatal", "política de preços", "politica de precos", "tributo", "imposto", "aneel", "anp"
    ],
    "cambio": ["dólar", "dolar", "câmbio", "cambio", "real", "moeda", "fx", "usd", "brl"],
    "dividendos": ["dividendo", "dividendos", "proventos", "jcp", "payout", "yield"],
    "resultado": [
        "balanço", "balanco", "resultado", "lucro", "ebitda", "receita", "guidance",
        "produção recorde", "producao recorde", "earnings"
    ],
}

# Termos que ajudam a dizer se o impacto tende a ser positivo ou negativo para a ação.
# Note que isso é relativo à PETR4 e não um sentimento genérico do texto.
SINAIS_IMPACTO = {
    "positivo": [
        "alta do petróleo", "alta do petroleo", "brent sobe", "corte de oferta", "dividendos",
        "lucro", "recorde de produção", "recorde de producao", "aumento de produção",
        "aumento de producao", "guidance positivo", "recompra", "descoberta"
    ],
    "negativo": [
        "queda do petróleo", "queda do petroleo", "brent cai", "intervenção do governo",
        "intervencao do governo", "controle de preços", "controle de precos", "acidente",
        "derramamento", "prejuízo", "prejuizo", "sanção", "sancao", "processo", "investigação",
        "investigacao", "redução de dividendos", "reducao de dividendos"
    ],
}

# Stopwords mínimas apenas para montar um ranking simples de palavras frequentes.
STOPWORDS = {
    "a", "o", "e", "de", "do", "da", "dos", "das", "em", "um", "uma", "para", "com",
    "por", "no", "na", "nos", "nas", "que", "os", "as", "ao", "à", "às", "ou", "se",
    "sobre", "mais", "menos", "entre", "após", "apos", "aponta", "diz", "segundo", "como",
    "yahoo", "finance", "reuters", "ap", "the", "and", "of", "in", "to", "for", "on", "at",
}


@dataclass
class NewsItem:
    title: str
    summary: str
    publisher: str
    link: str
    published_at: Optional[datetime]
    raw: Dict[str, Any]


# ==========================
# Utilidades de texto
# ==========================
def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[\w\-áàâãéèêíïóôõöúçñ]+", text, flags=re.IGNORECASE)
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def top_keywords_from_text(text: str, top_n: int = 12) -> List[str]:
    counts: Dict[str, int] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:top_n]]


# ==========================
# Busca de notícias Yahoo
# ==========================
def _safe_ts_to_datetime(value: Any) -> Optional[datetime]:
    try:
        if value is None:
            return None
        # Yahoo costuma trazer epoch seconds
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except Exception:
        return None


def _parse_news_item(item: Dict[str, Any]) -> Optional[NewsItem]:
    if not isinstance(item, dict):
        return None

    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = content.get("title") or item.get("title") or ""
    summary = content.get("summary") or item.get("summary") or ""
    publisher = content.get("provider") or content.get("publisher") or item.get("publisher") or ""

    link = ""
    canonical = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else None
    if canonical:
        link = canonical.get("url") or ""
    if not link:
        clickthrough = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else None
        if clickthrough:
            link = clickthrough.get("url") or ""
    if not link:
        link = content.get("link") or item.get("link") or ""

    published = (
        content.get("pubDate") or item.get("providerPublishTime") or item.get("pubDate")
    )
    published_at = _safe_ts_to_datetime(published)

    title = str(title).strip()
    summary = str(summary).strip()
    publisher = str(publisher).strip()
    link = str(link).strip()

    if not title:
        return None

    return NewsItem(
        title=title,
        summary=summary,
        publisher=publisher,
        link=link,
        published_at=published_at,
        raw=item,
    )


def fetch_yahoo_news(ticker: str, count: int = 20) -> List[NewsItem]:
    tk = yf.Ticker(ticker)
    raw_candidates: List[Dict[str, Any]] = []
    errors: List[str] = []

    # Tenta primeiro get_news, depois .news
    for method_name in ("get_news", "news"):
        try:
            if method_name == "get_news":
                method = getattr(tk, "get_news", None)
                if callable(method):
                    result = method(count=count, tab="news")
                else:
                    continue
            else:
                result = getattr(tk, "news", None)

            if isinstance(result, list) and result:
                raw_candidates = result
                break
        except Exception as exc:
            errors.append(f"{method_name}: {exc}")

    parsed: List[NewsItem] = []
    seen: set[Tuple[str, str]] = set()

    for item in raw_candidates:
        parsed_item = _parse_news_item(item)
        if not parsed_item:
            continue
        dedupe_key = (parsed_item.title.lower(), parsed_item.link.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        parsed.append(parsed_item)

    if not parsed and errors:
        raise RuntimeError(
            "Falha ao obter notícias do Yahoo Finance via yfinance. "
            f"Detalhes: {' | '.join(errors)}"
        )

    return parsed[:count]


# ==========================
# Relevância e impacto
# ==========================
def extract_themes(text: str) -> List[str]:
    text_n = normalize_text(text)
    found: List[str] = []
    for tema, palavras in TEMAS.items():
        if any(p in text_n for p in palavras):
            found.append(tema)
    return found


def extract_entities_simple(text: str, ticker: str) -> List[str]:
    text_n = normalize_text(text)
    entities = []
    perfil = ACOES[ticker]
    for ent in perfil["entidades"] + perfil["empresa_aliases"]:
        if ent in text_n:
            entities.append(ent)
    return sorted(set(entities))


def relevance_score(text: str, ticker: str) -> Tuple[int, List[str], List[str], List[str]]:
    text_n = normalize_text(text)
    perfil = ACOES[ticker]

    themes = extract_themes(text_n)
    entities = extract_entities_simple(text_n, ticker)
    top_keywords = top_keywords_from_text(text_n)

    score = 0

    # menção direta da empresa pesa mais
    for alias in perfil["empresa_aliases"]:
        if alias in text_n:
            score += 10
            break

    # entidade relacionada ao negócio
    for ent in perfil["entidades"]:
        if ent in text_n:
            score += 4

    # tema relevante para o ativo
    for theme in themes:
        if theme in perfil["temas"]:
            score += 3

    # bônus para combinações fortes
    if "petroleo" in themes and "geopolitica" in themes:
        score += 4
    if "governo" in themes and any(a in text_n for a in perfil["empresa_aliases"]):
        score += 5
    if "dividendos" in themes and any(a in text_n for a in perfil["empresa_aliases"]):
        score += 5
    if "resultado" in themes and any(a in text_n for a in perfil["empresa_aliases"]):
        score += 5

    return score, themes, entities, top_keywords


def classify_relevance(score: int) -> str:
    if score >= 18:
        return "alta"
    if score >= 9:
        return "media"
    return "baixa"


def classify_impact(text: str, ticker: str, is_relevant: bool) -> Tuple[str, float, List[str], List[str]]:
    """
    Classifica impacto provável para o ativo, não sentimento genérico do texto.
    Retorna: label, confidence, positive_hits, negative_hits
    """
    text_n = normalize_text(text)

    positive_hits = [term for term in SINAIS_IMPACTO["positivo"] if term in text_n]
    negative_hits = [term for term in SINAIS_IMPACTO["negativo"] if term in text_n]

    # heurísticas adicionais focadas em PETR4
    themes = extract_themes(text_n)
    if ticker == "PETR4.SA":
        if "petroleo" in themes and any(k in text_n for k in ["sobe", "alta", "dispara", "avança", "avanca"]):
            positive_hits.append("petróleo em alta")
        if "petroleo" in themes and any(k in text_n for k in ["cai", "queda", "recuo"]):
            negative_hits.append("petróleo em queda")
        if "governo" in themes and any(k in text_n for k in ["intervenção", "intervencao", "controle"]):
            negative_hits.append("pressão governamental")
        if "dividendos" in themes:
            positive_hits.append("tema dividendos")
        if "resultado" in themes and any(k in text_n for k in ["lucro", "recorde", "supera", "forte"]):
            positive_hits.append("resultado forte")
        if "resultado" in themes and any(k in text_n for k in ["prejuízo", "prejuizo", "fraco", "abaixo"]):
            negative_hits.append("resultado fraco")

    pos = len(set(positive_hits))
    neg = len(set(negative_hits))

    if not is_relevant:
        return "incerto", 0.15, sorted(set(positive_hits)), sorted(set(negative_hits))

    if pos == 0 and neg == 0:
        return "neutro", 0.35, [], []
    if pos > neg:
        conf = min(0.55 + (pos - neg) * 0.1, 0.9)
        return "positivo", conf, sorted(set(positive_hits)), sorted(set(negative_hits))
    if neg > pos:
        conf = min(0.55 + (neg - pos) * 0.1, 0.9)
        return "negativo", conf, sorted(set(positive_hits)), sorted(set(negative_hits))
    return "incerto", 0.4, sorted(set(positive_hits)), sorted(set(negative_hits))


# ==========================
# Pipeline principal
# ==========================
def analyze_news_for_ticker(ticker: str, count: int, query: Optional[str] = None) -> pd.DataFrame:
    if ticker not in ACOES:
        raise ValueError(f"Ticker '{ticker}' não está configurado no dicionário ACOES.")

    news_items = fetch_yahoo_news(ticker=ticker, count=count)
    rows: List[Dict[str, Any]] = []

    query_n = normalize_text(query) if query else ""

    for item in news_items:
        full_text = f"{item.title}. {item.summary}".strip()
        score, themes, entities, keywords = relevance_score(full_text, ticker)
        relevance = classify_relevance(score)

        # filtro opcional por query livre
        if query_n and query_n not in normalize_text(full_text):
            if relevance == "baixa":
                continue

        impact, confidence, pos_hits, neg_hits = classify_impact(
            full_text,
            ticker=ticker,
            is_relevant=(relevance != "baixa"),
        )

        rows.append({
            "ticker": ticker,
            "empresa": ACOES[ticker]["nome"],
            "published_at_utc": item.published_at.isoformat() if item.published_at else None,
            "publisher": item.publisher,
            "title": item.title,
            "summary": item.summary,
            "link": item.link,
            "relevance_score": score,
            "relevance_label": relevance,
            "impact_label": impact,
            "impact_confidence": round(confidence, 3),
            "themes": ", ".join(themes),
            "entities": ", ".join(entities),
            "top_keywords": ", ".join(keywords),
            "positive_signals": ", ".join(pos_hits),
            "negative_signals": ", ".join(neg_hits),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ordenação: relevância primeiro, depois data mais nova
    df["published_dt"] = pd.to_datetime(df["published_at_utc"], utc=True, errors="coerce")
    df = df.sort_values(
        by=["relevance_score", "published_dt"],
        ascending=[False, False],
        kind="mergesort",
    ).drop(columns=["published_dt"])
    return df.reset_index(drop=True)


# ==========================
# CLI
# ==========================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Busca notícias no Yahoo Finance e classifica relevância/impacto para PETR4."
    )
    parser.add_argument("--ticker", default="PETR4.SA", help="Ticker do Yahoo Finance. Padrão: PETR4.SA")
    parser.add_argument("--count", type=int, default=20, help="Quantidade máxima de notícias. Padrão: 20")
    parser.add_argument("--query", default=None, help="Filtro textual opcional para afunilar resultados")
    parser.add_argument(
        "--output",
        default="petr4_news_analysis.csv",
        help="Arquivo CSV de saída. Padrão: petr4_news_analysis.csv",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Arquivo JSON opcional com os resultados completos",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        df = analyze_news_for_ticker(
            ticker=args.ticker,
            count=max(1, args.count),
            query=args.query,
        )
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("Nenhuma notícia retornada ou classificada para os parâmetros informados.")
        return 0

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"CSV salvo em: {args.output}")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        print(f"JSON salvo em: {args.json_output}")

    cols_to_show = [
        "published_at_utc", "publisher", "title", "relevance_score",
        "relevance_label", "impact_label", "impact_confidence", "themes"
    ]
    print("\nResumo:\n")
    print(df[cols_to_show].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
