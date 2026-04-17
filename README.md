# Generic News Sentiment API

Projeto que recebe **um JSON de empresa** e **um JSON de notícia** e devolve o **sentimento da notícia relativo à empresa**.

## Estrutura

```text
news_sentiment_generic/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── sentiment.py
├── examples/
│   ├── company_petrobras.json
│   ├── news_oil.json
│   └── request_example.json
├── Dockerfile
├── README.md
├── requirements.txt
└── run_local.py
```

## Como funciona

1. Recebe `company` e `news`
2. Calcula um **sentimento base** com VADER
3. Calcula um **score de relevância** entre a notícia e a empresa
4. Infere um **contexto setorial** pela descrição da empresa
5. Aplica **ajustes setoriais** por tema e frases
6. Retorna um **score final entre -1 e 1** e um rótulo (`positivo`, `neutro`, `negativo`)

## Rodando localmente

```bash
pip install -r requirements.txt
python run_local.py
```

## Subindo a API

```bash
uvicorn app.main:app --reload
```

Acesse:
- `GET /health`
- `POST /analyze`
- docs Swagger em `http://127.0.0.1:8000/docs`

## Exemplo de request

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d @examples/request_example.json
```

## Exemplo de resposta

```json
{
  "ticker": "PETR4.SA",
  "company_name": "Petróleo Brasileiro S.A. - Petrobras",
  "base_sentiment_score": 0.25,
  "relevance_score": 0.76,
  "adjustment_score": 0.19,
  "final_sentiment_score": 0.44,
  "sentiment_label": "positivo",
  "confidence": 0.842,
  "detected_topics": ["oil", "geopolitics"],
  "positive_signals": ["topic:oil", "phrase:oil prices rise"],
  "negative_signals": [],
  "explanation": "Impacto positivo para Petróleo Brasileiro S.A. - Petrobras..."
}
```

## Docker

```bash
docker build -t generic-news-sentiment .
docker run -p 8000:8000 generic-news-sentiment
```

## Próximos passos

- trocar heurísticas por modelo treinado em pares `notícia + empresa`
- buscar perfil da empresa automaticamente de uma API externa
- plugar sua API de notícias antes do endpoint `/analyze`
- persistir resultados em banco
