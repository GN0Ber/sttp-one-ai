from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class CompanyInput(BaseModel):
    ticker: str = Field(..., examples=["PETR4.SA"])
    name: str = Field(..., examples=["Petróleo Brasileiro S.A. - Petrobras"])
    sector: Optional[str] = Field(default=None, examples=["Energy"])
    industry: Optional[str] = Field(default=None, examples=["Oil & Gas Integrated"])
    country: Optional[str] = Field(default=None, examples=["Brazil"])
    description: str = Field(..., examples=["Company focused on oil exploration, production, refining and fuel distribution."])
    extra: Dict[str, Any] = Field(default_factory=dict)


class NewsInput(BaseModel):
    title: str
    summary: Optional[str] = ""
    publisher: Optional[str] = None
    published_at: Optional[str] = None
    url: Optional[HttpUrl] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    company: CompanyInput
    news: NewsInput


class AnalyzeResponse(BaseModel):
    ticker: str
    company_name: str
    base_sentiment_score: float
    relevance_score: float
    adjustment_score: float
    final_sentiment_score: float
    sentiment_label: str
    confidence: float
    detected_topics: List[str]
    positive_signals: List[str]
    negative_signals: List[str]
    explanation: str
