# Test Metal ⚙️

Linear regression analysis and Pareto optimization framework for physicochemical properties of steel, with a modern web application for interactive use.

## Features

- **Linear Regression** — OLS-based analysis of element relationships in steel composition
- **Pareto Optimization** — Multi-objective optimization to find optimal input/output trade-offs
- **Web Dashboard** — React SPA with interactive Pareto charts and dataset management
- **REST API** — FastAPI backend with JWT auth, rate limiting, and OpenAPI docs
- **Async Tasks** — Celery workers for background regression/pipeline processing
- **Object Storage** — MinIO/S3-compatible storage for Excel uploads
- **Streamlit MVP** — Alternative lightweight UI for quick experimentation

## Architecture

```
                    ┌──────────────┐
                    │  React SPA   │  :80 / :3000
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼───┐ ┌─────▼──────┐ ┌───▼────────┐
     │  FastAPI   │ │  Celery    │ │  Streamlit │  :8501
     │  REST API  │ │  Workers   │ │  Dashboard │
     └─────┬──────┘ └─────┬──────┘ └────────────┘
           │              │
     ┌─────┴──────────────┴──────┐
     │     PostgreSQL + MinIO    │
     └───────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (optional, for full stack)

### Backend

```bash
cd web/backend
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env   # edit as needed
alembic upgrade head
uvicorn app.main:app --reload
```

### Frontend

```bash
cd web/frontend
npm install
npm run dev            # starts on :3000, proxies /api to :8000
```

### Docker (full stack)

```bash
cd web
docker compose up -d
```

This starts: PostgreSQL, Redis, MinIO, FastAPI, Celery worker/beat, Flower, React SPA (via nginx), and Streamlit.

## API Documentation

Once the backend is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

### Main Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/register` | Register user |
| POST | `/api/v1/auth/login` | Sign in |
| POST | `/api/v1/auth/refresh` | Refresh JWT |
| GET | `/api/v1/auth/me` | Current user |
| POST | `/api/v1/datasets` | Upload Excel |
| GET | `/api/v1/datasets` | List datasets |
| GET | `/api/v1/datasets/{id}` | Dataset detail |
| PATCH | `/api/v1/datasets/{id}` | Update dataset |
| DELETE | `/api/v1/datasets/{id}` | Delete dataset |
| POST | `/api/v1/datasets/{id}/regressions` | Run regression analysis |
| GET | `/api/v1/datasets/{id}/regressions` | List regression models |
| GET | `/api/v1/regressions/{id}` | Regression detail |
| POST | `/api/v1/optimizations` | Create Pareto optimization |
| GET | `/api/v1/optimizations` | List optimizations |
| GET | `/api/v1/optimizations/{id}` | Optimization detail |
| GET | `/api/v1/optimizations/{id}/points` | Pareto points |
| DELETE | `/api/v1/optimizations/{id}` | Delete optimization |
| GET | `/api/v1/tasks` | List async tasks |
| GET | `/api/v1/tasks/{id}` | Task status |
| GET | `/api/v1/reports/regression/{id}.csv` | Export regression CSV |
| GET | `/api/v1/reports/optimization/{id}.csv` | Export optimization CSV |
| WS | `/ws/tasks/{task_id}` | Real-time task progress |

## Project Structure

```
├── test_metal/              # Core library
│   ├── core/                #  Regression & optimization engines
│   ├── io/                  #  Excel/PDF/report generation
│   ├── pipeline.py          #  End-to-end analysis pipeline
│   └── config.py            #  Pipeline configuration
├── tests/                   # Core library tests
├── web/
│   ├── backend/             # FastAPI + Celery + Alembic
│   │   ├── app/
│   │   │   ├── api/         # Route handlers
│   │   │   ├── core/        # Config, security, database
│   │   │   ├── domain/      # SQLAlchemy models
│   │   │   ├── infrastructure/  # S3, Celery, repositories
│   │   │   ├── schemas/     # Pydantic models
│   │   │   ├── services/    # Business logic
│   │   │   └── workers/     # Celery tasks
│   │   ├── alembic/         # Database migrations
│   │   └── tests/           # Backend tests
│   ├── frontend/            # React + TypeScript SPA
│   │   └── src/
│   │       ├── api/         # HTTP client + endpoints
│   │       ├── components/  # Layout, guards
│   │       ├── contexts/    # Auth context
│   │       └── pages/       # All route pages
│   ├── streamlit/           # Streamlit MVP dashboard
│   ├── docker-compose.yml
│   └── .env.example
└── pyproject.toml
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.13, FastAPI, SQLAlchemy 2.0, Alembic |
| Frontend | React 19, TypeScript, Vite, Recharts |
| Dashboard | Streamlit, Plotly |
| Database | PostgreSQL |
| Cache/Queue | Redis, Celery |
| Storage | MinIO (S3-compatible) |
| Auth | JWT (python-jose), bcrypt |
| Monitoring | Prometheus metrics, Flower |
| Container | Docker, docker compose |

## License

MIT License

Copyright (c) 2024
