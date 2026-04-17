# PETR4 Sentiment V1

## Estrutura de pastas

```text
petr4_sentiment_v1_package/
├── app/
│   ├── __init__.py
│   ├── engine.py
│   └── main.py
├── docker/
│   └── Dockerfile
├── notebooks/
│   └── petr4_sentiment_v1_step_by_step.ipynb
├── scripts/
│   └── run_sentiment_cli.py
├── requirements.txt
└── README.md
```

## O que cada arquivo faz

- `app/engine.py`: motor principal de análise de notícias e sentimento.
- `app/main.py`: API FastAPI para uso estilo produção.
- `scripts/run_sentiment_cli.py`: execução via terminal e exportação para CSV/JSON.
- `notebooks/petr4_sentiment_v1_step_by_step.ipynb`: notebook para testes por etapa no Jupyter/Anaconda.
- `docker/Dockerfile`: imagem Docker da API.
- `requirements.txt`: dependências do projeto.

## Instalação local

```bash
pip install -r requirements.txt
```

## Rodar no terminal

Na raiz do projeto:

```bash
python scripts/run_sentiment_cli.py --ticker PETR4.SA --count 20
```

## Rodar API local

Na raiz do projeto:

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `GET /health`
- `GET /config/tickers`
- `GET /news/PETR4.SA?count=10`
- `POST /analyze-text`

### Exemplo de body para `/analyze-text`

```json
{
  "ticker": "PETR4.SA",
  "text": "Alta do petróleo e corte de produção da OPEP favorecem petroleiras."
}
```

## Docker

Na raiz do projeto:

```bash
docker build -f docker/Dockerfile -t petr4-sentiment-v1 .
docker run -p 8000:8000 petr4-sentiment-v1
```

## Observação

O notebook foi mantido separado em `notebooks/` para você executar passo a passo no Anaconda/Jupyter.
