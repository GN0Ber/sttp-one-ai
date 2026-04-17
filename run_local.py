import json
from pathlib import Path

from app.sentiment import analyze_news_for_company

BASE = Path(__file__).resolve().parent
company = json.loads((BASE / "examples" / "company_petrobras.json").read_text(encoding="utf-8"))
news = json.loads((BASE / "examples" / "news_oil.json").read_text(encoding="utf-8"))

result = analyze_news_for_company(company=company, news=news)
print(json.dumps(result, ensure_ascii=False, indent=2))
