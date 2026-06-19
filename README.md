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

## Core Library Usage

### Installation

```bash
# Clone and install
pip install -e ".[dev]"
```

### Quick Start (CLI)

Run the full pipeline on your Excel data:

```bash
# For .xlsx files (openpyxl engine)
python main.py --file source_data.xlsx --output outputs

# For .xls files (xlrd engine), override usecols to match sheet
python main.py --file source_data.xls --output outputs --usecols "A:ZZ"

# Run with specific predictors and target
python main.py --file data.xlsx --output results \
  --x-columns steel_S_before steel_Si_before \
  --y-column steel_S_after
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | `source_data.xls` | Path to Excel file |
| `--output` | `outputs` | Output directory |
| `--report` | `outputs/regression_report.csv` | Regression report path |
| `--mode` | `after` | Analysis mode: `after` or `before` |
| `--x-columns` | (auto) | Predictor columns (space-separated) |
| `--y-column` | (auto) | Target column |
| `--missing-threshold` | `0.5` | Max fraction of missing values per column |
| `--header-row` | `3` | Row number containing column headers |
| `--usecols` | `B:CN` | Column range to read from Excel |

### Programmatic Usage

```python
from pathlib import Path
from test_metal.config import ProjectConfig
from test_metal.pipeline import run_pipeline_with_io

cfg = ProjectConfig(
    excel_header_row=3,
    excel_usecols="B:CN",
    missing_threshold=0.5,
    outputs_dir=Path("outputs"),
)

result = run_pipeline_with_io(
    Path("source_data.xlsx"),
    config=cfg,
    x_columns=["steel_S_before", "steel_Si_before"],
    y_column="steel_S_after",
)

# Access results
for model in result.models:
    print(f"{model.x_col} -> {model.y_col}: R²={model.r2:.3f}")

# Optimization reports
if result.single_element_report is not None:
    print(result.single_element_report)

if result.pareto_front is not None:
    print(result.pareto_front)
```

### Excel Format Notes

- **`.xlsx` files** use the `openpyxl` engine (default).
- **`.xls` files** (older Excel format) require the `xlrd` engine. Install it and pass the correct `--usecols` range matching your sheet width.
- Column headers must be on the row specified by `--header-row` (default: row 3).
- The pipeline renames columns using the internal `COLUMN_NAMES` mapping.

### Outputs

After running, the `--output` directory contains:
- `regression_report.csv` — All OLS model coefficients and statistics
- `optimization_report_single_element.csv` — Inverse optimization results
- `optimization_report_pareto_front.csv` — Pareto-optimal solutions
- `all_regressions.pdf` — Combined regression plots
- Individual `.png` plots for each regression
- `run.log` — Execution log

### Running Tests

```bash
# All tests (core library)
pytest

# With coverage
pytest --cov=test_metal --cov-report=term-missing

# Backend tests (requires PostgreSQL + Redis + MinIO running)
cd web/backend
pytest

# Specific test file
pytest tests/test_regression.py -v
```

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

## Quick Start (Web App)

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (optional, for full stack)

### Docker (full stack)

```bash
cd web
docker compose up -d
```

This starts: PostgreSQL, Redis, MinIO, FastAPI, Celery worker/beat, Flower, React SPA (via nginx), and Streamlit.

### Windows Native Setup (without Docker)

Run all services directly on Windows without Docker.

#### 1. Infrastructure

**PostgreSQL** — Install locally (or use WSL). Update `DATABASE_URL` in `.env`.

**MinIO** — Download the executable and start:
```powershell
# Download minio.exe to C:\tools\minio\
# Start server
C:\tools\minio\minio.exe server C:\data\minio --console-address :9001
```

**Redis** — Use the Windows-native Redis 3.2.100 port:
```powershell
# Download Redis-x64-3.2.100 to C:\tools\redis\
C:\tools\redis\redis-server.exe
```
> **Note:** redis-py 8.0 defaults to RESP3, but Redis 3.2 only supports RESP2.  
> The app patches `redis.ConnectionPool` to force `protocol=2` automatically — no manual config needed.

#### 2. Backend

```powershell
cd web/backend
python -m venv .venv; .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env   # edit DB/Redis/S3 credentials as needed
alembic upgrade head
uvicorn app.main:app --reload
```

#### 3. Celery Worker

Start the Celery worker (requires `--pool=solo` on Windows, which lacks `os.fork()`):

```powershell
cd web/backend
.venv\Scripts\activate
celery -A app.infrastructure.task_queue worker --pool=solo -l info
```

#### 4. Frontend

```powershell
cd web/frontend
npm install
npm run dev            # starts on :3000, proxies /api to :8000
```

### Known Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `unknown command 'HELLO'` on Redis connect | redis-py 8.0 defaults to RESP3; Redis 3.2 only speaks RESP2 | `app/infrastructure/redis_compat.py` patches `ConnectionPool` to use `protocol=2` |
| `Connection._init_params() got an unexpected keyword argument 'protocol'` | `?protocol=2` in broker URL is not supported by kombu | Use the `redis_compat` monkey-patch instead of query params |
| Celery `NotImplementedError` on `os.fork()` | Windows lacks `fork()` | Start worker with `--pool=solo` |
| `TypeError: object Response can't be used in 'await' expression` | aiobotocore 3.7+ changed `create_client()` to return async context manager | Removed `await` before `_get_client()` calls in `storage.py` |
| Login returns 422 with form data | OAuth2PasswordRequestForm expects `application/x-www-form-urlencoded` | Changed to JSON body via `LoginRequest` schema |
| Frontend 401 loop on login failure | Auto-refresh interceptor retries failed `/auth/login` requests | Added `!path.startsWith("/auth/login")` guard in `client.ts` |

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
├── main.py                  # CLI entry point
├── web/
│   ├── backend/             # FastAPI + Celery + Alembic
│   │   ├── app/
│   │   │   ├── api/         # Route handlers
│   │   │   ├── core/        # Config, security, database
│   │   │   ├── domain/      # SQLAlchemy models
│   │   │   ├── infrastructure/  # S3, Celery, Redis, repositories
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
├── outputs/                 # Example pipeline outputs
├── source_data.xls          # Example data (old .xls format)
├── _test_source.xlsx        # Example data (.xlsx format)
└── pyproject.toml
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core Library | Python 3.13, statsmodels, scikit-learn, pandas |
| CLI | argparse, statsmodels OLS |
| Backend | FastAPI, SQLAlchemy 2.0, Alembic |
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
