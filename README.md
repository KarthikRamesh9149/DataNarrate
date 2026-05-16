# DataNarrate - Streamlit Data Science Copilot

Describe your analysis goal in plain English and get automated data profiling, cleaning, intent-aware visualizations, baseline ML model training, and plain-English summaries.

![DataNarrate](Icon.png)

## Highlights

- **Streamlit-first experience** - A polished single-page workflow for uploading data, analyzing quality, creating charts, training a baseline model, and exporting results.
- **Natural Language Intent Understanding** - Describe what you want to analyze; DataNarrate uses that intent to prioritize visualizations and summaries.
- **Smart Data Cleaning** - Apply safe or recommended cleaning with an auditable before/after log.
- **7 Auto-Generated Visualizations** - Generate de-duplicated charts with AI-powered planning when a Groq key is configured and rule-based fallbacks when it is not.
- **Baseline ML Model Training** - Suggest a target from your intent, then train classification or regression models with metrics, feature importance, and diagnostic artifacts.
- **Executive Summary** - Produce a plain-English final answer with caveats suitable for non-technical stakeholders.
- **Export Options** - Download the current cleaned dataset, all charts as a ZIP, and the final summary.

## Tech Stack

- **Frontend/App**: Streamlit
- **Analytics Engine**: Python, Pandas, NumPy, Matplotlib, scikit-learn
- **Optional API Layer**: FastAPI backend remains available for integrations
- **LLM**: Groq API with robust rule-based fallbacks

## Demo

🎥 **Project Demo (Vimeo)**  
Watch a full walkthrough of DataNarrate, including intent-based analysis, auto visualizations, and ML training:  
[https://vimeo.com/1158871881](https://vimeo.com/1158871881?fl=ip&fe=ec)

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│              streamlit_app.py (Streamlit)           │
│  - Upload/sample data workflow                      │
│  - Profile, clean, visualize, model, summarize tabs │
│  - Downloads for cleaned data, charts, summaries    │
└───────────────────────┬─────────────────────────────┘
                        │ imports/reuses
┌───────────────────────▼─────────────────────────────┐
│          datanarrate_server.py analytics engine      │
│  - Pandas data processing                            │
│  - Matplotlib visualization                          │
│  - scikit-learn ML pipeline                          │
│  - Groq LLM integration + fallbacks                  │
│  - Optional FastAPI endpoints                        │
└─────────────────────────────────────────────────────┘
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DataNarrate.git
cd DataNarrate
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure optional LLM access

DataNarrate works without an LLM by using rule-based fallbacks. For AI-generated chart plans and richer summaries, configure Groq keys:

```bash
cp .env.example .env
```

Edit `.env` and add one or more keys:

```env
GROQ_API_KEY_1=your_groq_api_key_here
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`. The repository includes `.streamlit/config.toml` so local and hosted runs use the same polished theme by default.

## Usage

1. **Load a dataset** - Upload CSV/Excel/JSON or load the bundled AFCON sample.
2. **Describe your analysis intent** - Enter the business or research question in the sidebar.
3. **Review data quality** - Inspect column types, missingness, uniqueness, and outlier indicators.
4. **Clean your data** - Apply safe or recommended cleaning and review the cleaning log.
5. **Generate visuals** - Create seven intent-aware charts and download them as a ZIP.
6. **Train a baseline model** - Ask DataNarrate to suggest a target from your intent, confirm the column, choose classification or regression, and review metrics and feature importance.
7. **Generate a final summary** - Produce and download a plain-English executive summary.

## Optional FastAPI mode

The original FastAPI service is still available for API integrations:

```bash
python datanarrate_server.py
```

The API runs on `http://localhost:8891` by default.

## Sample Dataset

Included: `afcon_2025_2026_dataset.csv` - Africa Cup of Nations match data for testing.

## License

MIT
