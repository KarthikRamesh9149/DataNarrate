# DataNarrate - Intent-Aware Data Science Copilot

Describe your analysis goal in plain English and get automated data profiling, cleaning, visualizations, and ML model training.

![DataNarrate](Icon.png)

## Features

- **Natural Language Intent Understanding** - Describe what you want to analyze
- **Smart Data Cleaning** - Automatic cleaning with before/after comparison
- **7 Auto-Generated Visualizations** - Charts with AI-powered explanations
- **ML Model Training** - Classification & regression with scikit-learn
- **Target Column Inference** - Smart detection with confidence scores
- **Export Options** - Download cleaned data (CSV) and charts (ZIP)

## Tech Stack

- **Frontend**: React/TypeScript (runs in ContextUI/Electron)
- **Backend**: Python FastAPI
- **ML/Data**: Pandas, NumPy, Matplotlib, scikit-learn
- **LLM**: Groq API (Llama 3.3 70B) with rule-based fallbacks

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           DataNarrateWindow.tsx (React)             │
│                       │                             │
│            HTTP REST API (localhost:8891)           │
└───────────────────────┼─────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────┐
│                       ▼                             │
│       datanarrate_server.py (FastAPI)               │
│   - Pandas data processing                          │
│   - Matplotlib visualization                        │
│   - scikit-learn ML pipeline                        │
│   - Groq LLM integration                            │
└─────────────────────────────────────────────────────┘
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DataNarrate.git
cd DataNarrate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key(s):

```
GROQ_API_KEY_1=your_groq_api_key_here
```

### 4. Run the backend server

```bash
python datanarrate_server.py
```

The server runs on `http://localhost:8891` by default.

### 5. Frontend

The frontend (`DataNarrateWindow.tsx`) is designed to run within ContextUI. Load it as a workflow window in your ContextUI installation.

## Usage

1. **Upload a dataset** (CSV or Excel)
2. **Describe your analysis intent** in natural language
3. **Review the data profile** - statistics, missing values, column types
4. **Clean your data** - automatic cleaning with detailed log
5. **Explore visualizations** - 7 auto-generated charts with explanations
6. **Train ML models** - select a target column and train
7. **Export results** - download cleaned data and charts

## Sample Dataset

Included: `afcon_2025_2026_dataset.csv` - Africa Cup of Nations match data for testing.

## License

MIT
