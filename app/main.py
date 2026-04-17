from fastapi import FastAPI

from app.models import AnalyzeRequest, AnalyzeResponse
from app.sentiment import analyze_news_for_company

app = FastAPI(
    title="Generic News Sentiment API",
    version="1.0.0",
    description="Recebe um JSON de empresa e um JSON de notícia e devolve sentimento da notícia relativo à empresa.",
)

@app.get("/")
def root() -> dict:
    return {
        "message": "Generic News Sentiment API online",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    result = analyze_news_for_company(
        company=payload.company.model_dump(),
        news=payload.news.model_dump(),
    )
    return AnalyzeResponse(**result)
