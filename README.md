# DataNarrate

[![CI](https://github.com/KarthikRamesh9149/DataNarrate/actions/workflows/ci.yml/badge.svg)](https://github.com/KarthikRamesh9149/DataNarrate/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20to%203.14-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

DataNarrate is an intent-aware data science copilot that turns an uploaded
dataset and a plain-English question into a complete exploratory analysis
workflow: profiling, cleaning, visualization, baseline modeling, and an
executive-ready summary.

![DataNarrate app icon](Icon.png)

The project is designed as a local-first analytics tool. The Streamlit app gives
analysts and non-technical stakeholders a guided interface, while the reusable
Python engine exposes the same capabilities through optional FastAPI endpoints.
It can use Groq-hosted LLMs for chart planning and explanations, but every major
workflow has rule-based fallbacks so the app still works without an API key.

## Why This Matters

Most one-off data questions stall before the interesting work begins: messy
columns, missing values, unclear targets, chart selection, and modeling boilerplate.
DataNarrate compresses that setup into a reviewable workflow. It does not try to
replace statistical judgment. Instead, it gives a fast, transparent first pass
with audit logs, downloadable artifacts, model caveats, and conservative language
around correlation and prediction.

## Product Highlights

- **Natural-language analysis intent** drives chart planning, target suggestions,
  summaries, and model setup.
- **Multi-format ingestion** supports CSV, XLSX, XLS, and JSON uploads, plus a
  bundled AFCON sample dataset.
- **Automated profiling** reports shape, column types, missingness, uniqueness,
  duplicate rows, and IQR-based outlier indicators.
- **Two-stage cleaning** applies safe normalization by default and optional
  recommended cleaning for imputation, outlier clipping, and rare category grouping.
- **Seven-chart visualization workflow** generates diverse, de-duplicated
  Matplotlib PNG charts with LLM-assisted planning when configured.
- **Baseline ML training** supports classification and regression with
  scikit-learn pipelines, metrics, diagnostic plots, and ranked coefficients.
- **Executive summary generation** combines profile, cleaning, charts, and model
  context into a plain-English final answer.
- **Exportable artifacts** include cleaned CSVs, chart ZIPs, summary text, and
  model plots.
- **Optional API mode** keeps FastAPI endpoints available for automation and
  integrations.

## Demo

A walkthrough link is included in the repository history:
[Vimeo demo](https://vimeo.com/1158871881?fl=ip&fe=ec).

## Architecture

```text
DataNarrate/
  streamlit_app.py         Streamlit UI and session workflow
        |
        | in-process calls
        v
  datanarrate_server.py    Analytics engine plus optional FastAPI app
        |
        +-- pandas/numpy profiling and cleaning
        +-- matplotlib chart generation
        +-- scikit-learn preprocessing and baseline models
        +-- Groq LLM calls with model/key fallbacks
        +-- local filesystem artifacts under outputs/
```

The Streamlit app imports `datanarrate_server.py` directly rather than requiring
a separate local HTTP service. That keeps the default experience simple:

1. Load a dataset in the sidebar.
2. Capture the user's analysis intent.
3. Profile the active DataFrame.
4. Apply safe or recommended cleaning.
5. Generate a seven-chart plan and chart images.
6. Infer or select a target column.
7. Train a baseline classifier or regressor.
8. Produce a final stakeholder summary.

The FastAPI app remains in the same module for clients that need an HTTP API.
State is stored in memory for the active process, and generated files are written
under `outputs/`.

## Technical Depth

### Data Profiling

`profile_dataframe()` inspects every column and returns:

- numeric, categorical, and datetime column groups;
- missing counts and percentages;
- unique counts and constant-column indicators;
- numeric min, max, mean, standard deviation, and outlier percentage;
- categorical top values;
- duplicate row counts and a compact profile summary.

### Cleaning Pipeline

Safe cleaning is intended to be low-risk and auditable:

- normalize column names to `snake_case`;
- trim text values;
- standardize common missing tokens such as `N/A`, `null`, `--`, and `missing`;
- coerce likely numeric and datetime columns;
- remove exact duplicate rows;
- record each action in a cleaning log.

Recommended cleaning is user-triggered and more opinionated:

- median imputation for numeric columns;
- mode imputation for categorical columns;
- winsorization at the 1st and 99th percentiles for outlier-heavy numeric columns;
- grouping rare categories under `Other`.

### Visualization Planning

The chart workflow asks the LLM for a plan when Groq credentials are available.
If the LLM is unavailable, the engine falls back to a deterministic plan based on
the profile. Column translation and chart signatures reduce failures caused by
renamed columns and skip duplicate chart specs.

Supported chart families include missing-value bars, numeric distributions,
categorical frequency bars, correlation heatmaps, scatter plots, boxplots,
time trends, and grouped aggregations.

### Modeling Pipeline

The modeling flow uses scikit-learn pipelines:

- `ColumnTransformer` separates numeric and categorical features;
- numeric features use median imputation and standard scaling;
- categorical features use most-frequent imputation and one-hot encoding;
- classification uses `LogisticRegression`;
- regression uses `Ridge`;
- task type is validated against the target data, not blindly trusted from intent;
- classification returns accuracy, macro F1, and a confusion matrix plot;
- regression returns RMSE, MAE, R-squared, and a residuals plot;
- coefficients are surfaced as baseline feature-importance signals.

These models are intentionally described as baselines. The app warns users about
leakage, sampling, validation, and the difference between model reliance and
real-world causation.

## Repository Structure

```text
.
|-- streamlit_app.py                 # Streamlit-first user interface
|-- datanarrate_server.py            # Analytics engine and optional FastAPI app
|-- tests/
|   `-- test_datanarrate_server.py   # Unit tests for config, cleaning, profiling
|-- .github/workflows/ci.yml         # Ruff, py_compile, and pytest matrix
|-- .streamlit/config.toml           # Local Streamlit theme/server defaults
|-- .env.example                     # Optional Groq/server configuration template
|-- requirements.txt                 # Runtime dependencies
|-- requirements-dev.txt             # Runtime plus pytest and ruff
|-- pyproject.toml                   # Package metadata, pytest, and ruff config
|-- ARCHITECTURE.md                  # Detailed architecture notes
|-- FEATURES_GUIDE.md                # Product feature reference
|-- afcon_2025_2026_dataset.csv      # Bundled sample dataset
|-- data/                            # Additional local/sample data
`-- outputs/                         # Generated files, ignored by git
```

## Quick Start

### 1. Clone and enter the repo

```bash
git clone https://github.com/KarthikRamesh9149/DataNarrate.git
cd DataNarrate
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development and CI parity:

```bash
python -m pip install -r requirements-dev.txt
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Configuration

DataNarrate works without environment variables. In that mode, LLM-assisted
planning and summaries fall back to deterministic logic.

To enable Groq-backed LLM features:

```bash
# macOS/Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Then fill in one or more keys:

```env
GROQ_API_KEY_1=
GROQ_API_KEY_2=
GROQ_API_KEY_3=
GROQ_API_KEY_4=
```

Supported environment variables:

- `GROQ_API_KEY_1` to `GROQ_API_KEY_4`: optional Groq API keys for
  key rotation. Default: empty.
- `GROQ_API_KEY`: legacy single-key fallback. Default: empty.
- `GROQ_BASE_URL`: OpenAI-compatible Groq API base URL. Default:
  `https://api.groq.com/openai/v1`.
- `PRIMARY_MODEL`: first model attempted for LLM calls. Default:
  `groq/compound`.
- `FALLBACK_MODEL_1`: second model attempted. Default:
  `llama-3.3-70b-versatile`.
- `FALLBACK_MODEL_2`: optional third model attempted. Default: empty.
- `LLM_TIMEOUT_SECONDS`: HTTP timeout for LLM requests. Default: `30`.
- `MAX_RETRIES`: parsed LLM retry setting. The current fallback path relies
  mainly on key/model rotation. Default: `1`.
- `SERVER_HOST`: optional FastAPI host. Default: `127.0.0.1`.
- `SERVER_PORT`: optional FastAPI port. Default: `8891`.
- `CORS_ORIGINS`: comma-separated origins for API CORS. Default: local
  Streamlit/API origins.

## Optional FastAPI Mode

Start the API server when another client needs HTTP access:

```bash
python datanarrate_server.py
```

Use a custom port by passing it as the first argument:

```bash
python datanarrate_server.py 8892
```

Core endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/status` | Health and current dataset status |
| `POST` | `/ingest` | Load uploaded data or the bundled sample |
| `POST` | `/profile` | Generate a data profile |
| `POST` | `/infer-target` | Suggest prediction targets from intent |
| `POST` | `/set-target` | Store an explicit target column |
| `POST` | `/plan` | Generate a visualization plan |
| `POST` | `/clean/apply` | Apply safe or recommended cleaning |
| `POST` | `/charts` | Generate chart images |
| `POST` | `/explain` | Explain generated charts |
| `POST` | `/cleaning-summary` | Compare original and cleaned data |
| `POST` | `/final-summary` | Generate the final narrative summary |
| `POST` | `/model/train` | Train a baseline model |
| `POST` | `/model/explain` | Explain model metrics and coefficients |
| `GET` | `/download/cleaned` | Download the latest cleaned CSV |
| `GET` | `/download/charts` | Download chart PNGs as a ZIP |
| `GET` | `/chart/{filename}` | Fetch a generated chart |
| `GET` | `/model/artifact/{filename}` | Fetch a model diagnostic plot |

## Common Workflows

### Analyze a Dataset in the UI

1. Upload CSV, Excel, or JSON data, or load the bundled AFCON sample.
2. Describe the question in plain English.
3. Review the Quality tab for data shape and quality issues.
4. Apply safe cleaning, or recommended cleaning when imputation/outlier handling
   is acceptable for the use case.
5. Generate a chart plan and charts.
6. Pick or infer a target column, then train a baseline model.
7. Download the cleaned dataset, chart ZIP, and final summary.

### Run Quality Checks Locally

```bash
python -m ruff check .
python -m py_compile streamlit_app.py datanarrate_server.py
python -m pytest
```

CI runs those checks on Python 3.10, 3.12, and 3.14 for pushes to `main` and
`codex/**`, plus pull requests targeting `main`.

## Security and Privacy Notes

- `.env` is ignored by git; keep API keys there rather than in source files.
- Streamlit mode reads uploaded files in memory and writes generated artifacts
  under `outputs/`.
- FastAPI upload mode saves uploaded datasets under `data/`.
- `outputs/` is ignored by git because it can contain cleaned datasets, charts,
  chart ZIPs, and model artifacts.
- LLM features send profile and analysis context to the configured Groq endpoint.
- API request models include `allow_sample_rows`; the Streamlit app currently
  calls LLM-backed workflows with sample rows disabled.
- The app uses conservative prompts and fallback summaries that avoid causal
  claims from exploratory charts or baseline coefficients.
- CORS defaults are local development origins. Set `CORS_ORIGINS` explicitly
  before exposing the optional API to another frontend.

## Troubleshooting

- `ModuleNotFoundError` on startup:
  dependencies were probably not installed in the active environment. Activate
  `.venv` and run `python -m pip install -r requirements.txt`.
- Excel upload fails:
  reinstall dependencies and confirm the file is `.xlsx` or `.xls`.
- LLM output falls back to templates:
  check `.env`, key quota, `GROQ_BASE_URL`, and configured model names.
- API blocked by browser CORS:
  add the frontend origin to the comma-separated `CORS_ORIGINS` value.
- Model training fails with target errors:
  choose another target or clean/impute first. This can happen when the target
  has too few non-null rows or no usable feature columns remain.
- Charts are missing or stale:
  regenerate charts or clear `outputs/charts/` locally.

## Roadmap and Limitations

Current limitations:

- App state is in memory and scoped to the running process.
- There is no database, authentication layer, or multi-user workspace model.
- Baseline models use linear scikit-learn estimators, not advanced AutoML.
- Generated charts are Matplotlib PNGs, not interactive dashboard components.
- LLM quality depends on configured Groq model availability and response quality.
- The app does not claim production-grade statistical validity; outputs require
  domain review before decisions are made.

Practical next steps:

- add persisted analysis sessions;
- add cross-validation and richer model diagnostics;
- expose a documented OpenAPI usage guide;
- add screenshot-based UI smoke tests;
- support interactive chart exports;
- expand test coverage around chart planning, model training, and API endpoints.

## Quality Signals

- Small, inspectable Python codebase with a Streamlit UI and reusable engine.
- Unit tests cover environment parsing, safe cleaning, and data profiling.
- Ruff and `py_compile` catch lint and syntax regressions.
- GitHub Actions runs the quality suite across multiple Python versions.
- Generated outputs and secrets are intentionally excluded from version control.
- Architecture and feature reference docs are included for maintainers.

## License

MIT. See the repository license metadata in `pyproject.toml`.
