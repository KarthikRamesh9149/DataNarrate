# DataNarrate - Technical Architecture Document

---

## System Overview

DataNarrate is a full-stack data science application with a React frontend and Python/FastAPI backend. It integrates LLM capabilities (via Groq API) for intelligent analysis while maintaining rule-based fallbacks for reliability.

```
┌─────────────────────────────────────────────────────────────────┐
│                     ContextUI (Electron App)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              DataNarrateWindow.tsx (React)                 │  │
│  │  - State management via React hooks                        │  │
│  │  - Fetch API for backend communication                     │  │
│  │  - Dynamic component (no imports, globals provided)        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                    HTTP REST API (localhost:8891)                │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              datanarrate_server.py (FastAPI)               │  │
│  │  - REST API endpoints                                      │  │
│  │  - Pandas data processing                                  │  │
│  │  - Matplotlib visualization                                │  │
│  │  - scikit-learn ML pipeline                                │  │
│  │  - Groq LLM integration                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│              ┌───────────────┴───────────────┐                   │
│              ▼                               ▼                   │
│     ┌─────────────┐                 ┌─────────────────┐          │
│     │  Local FS   │                 │   Groq API      │          │
│     │  - data/    │                 │   (External)    │          │
│     │  - outputs/ │                 │   LLM Calls     │          │
│     └─────────────┘                 └─────────────────┘          │
│                                                                  │
│                    Python Backend Server                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
DataNarrate/
├── datanarrate_server.py      # Main backend server (FastAPI)
├── DataNarrateWindow.tsx      # Frontend React component
├── .env                       # Environment configuration
├── .env.example               # Template for environment variables
├── afcon_2025_2026_dataset.csv/.xlsx  # Sample dataset
│
├── data/                      # Uploaded datasets stored here
│   └── [uploaded_files]
│
└── outputs/                   # Generated outputs
    ├── cleaned/               # Cleaned CSV files
    │   └── cleaned_[dataset].csv
    ├── charts/                # Generated PNG charts
    │   └── chart_[n]_[type].png
    └── modeling/              # Model artifacts
        ├── confusion_matrix.png
        └── residuals.png
```

---

## Backend Architecture

### FastAPI Application Structure

```python
# Core app initialization
app = FastAPI(title="DataNarrate Server", version="1.0.0")
app.add_middleware(CORSMiddleware, ...)  # Enable cross-origin requests

# Application State (in-memory)
app_state = {
    "df": None,              # Original DataFrame
    "df_cleaned": None,      # Cleaned DataFrame
    "dataset_name": None,    # Current dataset name
    "dataset_path": None,    # File path
    "profile": None,         # Data profile dict
    "cleaning_log": [],      # Cleaning actions
    "viz_plan": [],          # Visualization plan
    "chart_stats": {},       # Chart statistics
    "column_name_mapping": {},  # Original -> cleaned name mapping
    "target_column": None,   # Selected target for ML
    "target_candidates": [], # Possible target columns
    "model": None,           # Trained sklearn pipeline
    "model_results": None,   # Model metrics and artifacts
    "has_trained_model": False,
}
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/status` | GET | Health check, dataset status |
| `/ingest` | POST | Upload/load dataset |
| `/profile` | POST | Generate data quality profile |
| `/infer-target` | POST | LLM-driven target column inference |
| `/set-target` | POST | Explicitly set target column |
| `/plan` | POST | Generate visualization plan |
| `/clean/apply` | POST | Apply cleaning operations |
| `/cleaning-summary` | POST | Get before/after cleaning stats |
| `/charts` | POST | Generate 7 visualizations |
| `/explain` | POST | Generate chart explanations |
| `/final-summary` | POST | Generate comprehensive summary |
| `/model/train` | POST | Train ML model |
| `/model/explain` | POST | Explain model results |
| `/model/artifact/{filename}` | GET | Retrieve model plots |
| `/chart/{filename}` | GET | Retrieve chart images |
| `/download/cleaned` | GET | Download cleaned CSV |
| `/download/charts` | GET | Download charts ZIP |
| `/shutdown` | POST | Stop server |

---

## Data Flow Architecture

### 1. Data Ingestion Flow

```
User Action: Upload file / Use sample
                    │
                    ▼
        ┌───────────────────┐
        │   /ingest POST    │
        └─────────┬─────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────┐             ┌─────────────┐
│ Upload  │             │ Use Sample  │
│ File    │             │ Dataset     │
└────┬────┘             └──────┬──────┘
     │                         │
     └──────────┬──────────────┘
                ▼
    ┌───────────────────────────┐
    │  Parse file (pandas)      │
    │  - CSV: engine='python'   │
    │  - Excel: openpyxl        │
    │  - JSON: read_json        │
    └─────────────┬─────────────┘
                  ▼
    ┌───────────────────────────┐
    │  Reset ALL app_state      │
    │  Store df, dataset_name   │
    └─────────────┬─────────────┘
                  ▼
    ┌───────────────────────────┐
    │  Return: preview, shape,  │
    │  columns                  │
    └───────────────────────────┘
```

### 2. Analysis Flow

```
User Action: Click "Run Analysis"
                    │
                    ▼
        ┌───────────────────┐
        │  /profile POST    │ ◄── Step 1: Profile original data
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ profile_dataframe │
        │ - Column types    │
        │ - Missing counts  │
        │ - Outlier %       │
        │ - Duplicates      │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ /infer-target     │ ◄── Step 2: Determine analysis type
        │      POST         │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │  LLM Analysis     │
        │  - Parse intent   │
        │  - Match columns  │
        │  - Suggest target │
        │  - Detect task    │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Return candidates │
        │ with confidence   │
        └───────────────────┘
```

### 3. Cleaning Flow

```
User Action: Click "Apply Cleaning & Generate Charts"
                    │
                    ▼
        ┌───────────────────┐
        │ /clean/apply POST │
        └─────────┬─────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────────────┐     ┌─────────────────┐
│ Safe Cleaning   │     │ Recommended     │
│ (Always)        │     │ (If requested)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ - snake_case    │     │ - Median impute │
│ - Trim spaces   │     │ - Mode impute   │
│ - Normalize NaN │     │ - Winsorize     │
│ - Type coerce   │     │ - Group rare    │
│ - Deduplicate   │     │   categories    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Save to outputs/     │
         │  cleaned/             │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Re-profile cleaned   │
         │  data                 │
         └───────────────────────┘
```

### 4. Visualization Flow

```
        ┌───────────────────┐
        │    /plan POST     │ ◄── Generate viz plan via LLM
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │  LLM suggests 7   │
        │  chart specs      │
        │  OR               │
        │  Default fallback │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   /charts POST    │
        └─────────┬─────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  For each chart spec:       │
    │  1. Translate column names  │
    │  2. Generate signature      │
    │  3. Skip if duplicate       │
    │  4. Validate columns        │
    │  5. Generate matplotlib     │
    │  6. Save PNG                │
    │  7. Collect stats           │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Backfill to ensure 7       │
    │  unique charts              │
    └─────────────────────────────┘
```

### 5. Model Training Flow

```
User Action: Click "Train Model"
                    │
                    ▼
        ┌───────────────────┐
        │ /model/train POST │
        └─────────┬─────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Validate target column     │
    │  Drop rows with missing y   │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Smart Task Type Detection  │
    │  - numeric + <=10 unique    │
    │    → classification         │
    │  - numeric + >50% unique    │
    │    → regression             │
    │  - categorical              │
    │    → classification         │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Build sklearn Pipeline     │
    │  ┌─────────────────────┐    │
    │  │  ColumnTransformer  │    │
    │  │  - Numeric:         │    │
    │  │    Imputer→Scaler   │    │
    │  │  - Categorical:     │    │
    │  │    Imputer→OneHot   │    │
    │  └──────────┬──────────┘    │
    │             ▼               │
    │  ┌─────────────────────┐    │
    │  │  Model              │    │
    │  │  - Classification:  │    │
    │  │    LogisticRegress  │    │
    │  │  - Regression:      │    │
    │  │    Ridge            │    │
    │  └─────────────────────┘    │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Train/Test Split (80/20)   │
    │  Fit model                  │
    │  Predict on test set        │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Compute Metrics            │
    │  - Classification:          │
    │    accuracy, F1, confusion  │
    │  - Regression:              │
    │    R², RMSE, MAE            │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Generate Artifacts         │
    │  - Confusion matrix PNG     │
    │  - OR Residuals plot PNG    │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Extract Feature Importance │
    │  (coefficients ranked)      │
    └─────────────────────────────┘
```

---

## LLM Integration Architecture

### Multi-Key Capacity Handling

```python
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
]

# Fallback to legacy single key
if not GROQ_API_KEYS:
    GROQ_API_KEYS = [os.getenv("GROQ_API_KEY")]
```

### Model Fallback Chain

```
PRIMARY_MODEL (e.g., llama-3.3-70b-versatile)
        │
        ▼ (on error)
FALLBACK_MODEL_1 (e.g., llama3-70b-8192)
        │
        ▼ (on error)
FALLBACK_MODEL_2 (e.g., mixtral-8x7b-32768)
        │
        ▼ (on error)
Rule-based fallback (no LLM)
```

### LLM Call Flow

```python
async def call_llm(messages, json_mode=True):
    for model in [PRIMARY, FALLBACK_1, FALLBACK_2]:
        for api_key in GROQ_API_KEYS:
            try:
                response = await client.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 2000,
                        "response_format": {"type": "json_object"} if json_mode else None
                    }
                )
                if response.status_code == 200:
                    return parse_response(response)
                elif response.status_code == 429:
                    continue  # Try next key
                else:
                    break  # Try next model
            except Timeout:
                continue  # Try next key
    return None  # All failed, use rule-based fallback
```

### LLM Usage Points

| Endpoint | LLM Purpose | Fallback |
|----------|-------------|----------|
| `/infer-target` | Analyze intent, suggest target | Return all non-ID columns |
| `/plan` | Generate visualization plan | Default 7-chart plan |
| `/explain` | Generate chart explanations | Template-based explanation |
| `/final-summary` | Comprehensive analysis summary | Structured template |
| `/model/explain` | Explain model results | Metrics-based template |

---

## Frontend Architecture

### State Management

```typescript
// Core data state
const [datasetInfo, setDatasetInfo] = useState(null);
const [profile, setProfile] = useState(null);
const [charts, setCharts] = useState([]);
const [explanations, setExplanations] = useState([]);

// Target inference state
const [targetInference, setTargetInference] = useState(null);
const [selectedTarget, setSelectedTarget] = useState(null);
const [showTargetSelector, setShowTargetSelector] = useState(false);

// Modeling state
const [modelResults, setModelResults] = useState(null);
const [modelExplanation, setModelExplanation] = useState(null);
const [loadingModel, setLoadingModel] = useState(false);

// Server state
const [serverOnline, setServerOnline] = useState(false);
const [serverRunning, setServerRunning] = useState(false);

// UI state
const [activeTab, setActiveTab] = useState("setup");
const [analysisPhase, setAnalysisPhase] = useState("idle");
```

### Server Communication

```typescript
// All API calls use standard Fetch API
const runAnalysis = async () => {
  // Step 1: Profile
  const profileRes = await fetch(`${getServerUrl()}/profile`, { method: "POST" });

  // Step 2: Infer target
  const targetRes = await fetch(`${getServerUrl()}/infer-target`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ intent, allow_sample_rows: allowSampleRows }),
  });

  // ... etc
};
```

### Server Health Polling

```typescript
// Poll server status every 2 seconds
useEffect(() => {
  const checkStatus = async () => {
    try {
      const res = await fetch(`${getServerUrl()}/status`);
      const data = await res.json();
      setServerOnline(data.success);
    } catch {
      setServerOnline(false);
    }
  };

  checkStatus();
  const interval = setInterval(checkStatus, 2000);
  return () => clearInterval(interval);
}, [serverPort]);
```

### IPC Communication (Electron)

```typescript
// Start Python server via Electron IPC
const startServer = async () => {
  const ipcRenderer = window.require('electron').ipcRenderer;

  // Resolve script path
  const scriptResult = await ipcRenderer.invoke('resolve-workflow-script', {
    workflowFolder: 'DataNarrate',
    scriptName: 'datanarrate_server.py'
  });

  // Start server process
  const result = await ipcRenderer.invoke('python-start-script-server', {
    venvName: selectedVenv,
    scriptPath: scriptResult.path,
    port: serverPort,
    serverName: 'datanarrate',
  });
};
```

---

## Data Processing Architecture

### Column Name Translation System

```
Problem: LLM suggests columns using original names, but data uses cleaned names

Solution: Multi-layer fuzzy matching

┌─────────────────────────────────────────────────┐
│           translate_column_name()                │
├─────────────────────────────────────────────────┤
│ 1. Exact match from mapping dict                │
│ 2. Exact match in df columns                    │
│ 3. Snake_case version match                     │
│ 4. Case-insensitive match                       │
│ 5. Case-insensitive from mapping values         │
│ 6. Normalized string match (alphanumeric only)  │
│ 7. Substring/partial match                      │
│ 8. Word overlap scoring                         │
│ 9. Character similarity (Jaccard)               │
│ 10. Fallback to snake_case                      │
└─────────────────────────────────────────────────┘
```

### Chart Deduplication

```python
def get_chart_signature(chart_type: str, config: Dict) -> str:
    """Generate unique signature for chart deduplication."""
    cols = []
    for key in ['column', 'x_column', 'y_column', ...]:
        if key in config and config[key]:
            cols.append(config[key])
    cols_str = "_".join(sorted(set(cols)))
    agg = config.get('aggregation', '')
    return f"{chart_type}|{cols_str}|{agg}"

# Usage in chart generation
seen_signatures = set()
for chart_spec in viz_plan:
    signature = get_chart_signature(chart_type, config)
    if signature in seen_signatures:
        continue  # Skip duplicate
    seen_signatures.add(signature)
    # Generate chart...
```

---

## ML Pipeline Architecture

### Preprocessing Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder='drop')
```

### Full Pipeline Structure

```
┌──────────────────────────────────────────────────────┐
│                   sklearn Pipeline                    │
├──────────────────────────────────────────────────────┤
│  Step 1: preprocessor (ColumnTransformer)            │
│  ┌────────────────────────────────────────────────┐  │
│  │  Numeric Branch          Categorical Branch    │  │
│  │  ┌──────────────┐       ┌──────────────────┐   │  │
│  │  │SimpleImputer │       │ SimpleImputer    │   │  │
│  │  │(median)      │       │ (most_frequent)  │   │  │
│  │  └──────┬───────┘       └────────┬─────────┘   │  │
│  │         ▼                        ▼             │  │
│  │  ┌──────────────┐       ┌──────────────────┐   │  │
│  │  │StandardScaler│       │ OneHotEncoder    │   │  │
│  │  └──────────────┘       └──────────────────┘   │  │
│  └────────────────────────────────────────────────┘  │
│                          ▼                           │
│  Step 2: classifier/regressor                        │
│  ┌────────────────────────────────────────────────┐  │
│  │  LogisticRegression (classification)           │  │
│  │  OR                                            │  │
│  │  Ridge (regression)                            │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## Error Handling Strategy

### Backend Error Handling

```python
@app.post("/endpoint")
async def endpoint_handler(request: RequestModel):
    try:
        # Main logic
        return make_response(True, data={...})
    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))

def make_response(success: bool, data=None, error=None):
    return {"success": success, "data": data, "error": error}
```

### Frontend Error Handling

```typescript
const handleError = (message) => {
  setError(message);
  setTimeout(() => setError(null), 5000);  // Auto-dismiss after 5s
};

try {
  const res = await fetch(...);
  const data = await res.json();
  if (data.success) {
    // Handle success
  } else {
    handleError(data.error || "Operation failed");
  }
} catch (err) {
  handleError("Failed to connect to server");
}
```

### LLM Fallback Strategy

```
LLM Available                    LLM Unavailable
     │                                │
     ▼                                ▼
┌─────────────┐              ┌─────────────────┐
│ LLM-driven  │              │ Rule-based      │
│ response    │              │ fallback        │
└─────────────┘              └─────────────────┘
     │                                │
     ▼                                ▼
┌─────────────────────────────────────────────┐
│  Validate & sanitize output                  │
│  (both paths merge here)                     │
└─────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables (.env)

```bash
# Groq API Configuration
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
GROQ_API_KEY_4=gsk_...

# Legacy single key (fallback)
GROQ_API_KEY=gsk_...

# Groq API Base URL
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Model Configuration
PRIMARY_MODEL=llama-3.3-70b-versatile
FALLBACK_MODEL_1=llama3-70b-8192
FALLBACK_MODEL_2=mixtral-8x7b-32768

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8891
LLM_TIMEOUT_SECONDS=30
MAX_RETRIES=1
```

---

## Security Considerations

### Data Privacy
- All processing happens locally (no data sent to cloud except LLM)
- Optional: `allow_sample_rows` controls whether any data goes to LLM
- When disabled, only column names and statistics are sent to LLM

### LLM Safety Prompts
- Strict language rules prevent causal claims
- Mandatory disclaimers about exploratory nature
- Leakage detection warns about suspicious features
- No domain generalizations beyond dataset

### Input Validation
- Pydantic models validate all API inputs
- Column existence checked before operations
- File extension validation on upload
- SQL injection not applicable (no database)

---

## Performance Considerations

### Bottlenecks
1. **LLM Calls**: 10-30 second latency per call
2. **Chart Generation**: ~1-2 seconds per chart
3. **Model Training**: Varies with data size (typically <10 seconds)

### Optimizations
- Chart deduplication reduces redundant generation
- Profile caching (reused across endpoints)
- Async HTTP client for LLM calls
- Model fallback prevents total failure on LLM issues

### Memory Usage
- Full dataset held in memory (app_state["df"])
- Cleaned copy also in memory (app_state["df_cleaned"])
- Chart images saved to disk, not memory
- Model object retained for potential reuse

---

## Testing & Debugging

### Server Logs
```python
print(f"[Model] Target '{target}': {n_unique} unique values -> task_type={task_type}")
print(f"[Target Inference] LLM returned {len(candidates)} candidates")
print(f"LLM error ({model}): {status_code} - {response.text[:200]}")
```

### Health Check
```bash
curl http://localhost:8891/status
# Response: {"success": true, "data": {"status": "online", "dataset_loaded": true, ...}}
```

### Manual API Testing
```bash
# Profile current dataset
curl -X POST http://localhost:8891/profile

# Infer target
curl -X POST http://localhost:8891/infer-target \
  -H "Content-Type: application/json" \
  -d '{"intent": "What determines outcomes?", "allow_sample_rows": false}'
```

---

## Deployment Notes

### Requirements
- Python 3.8+ with virtual environment
- Node.js for ContextUI
- Groq API key for LLM features

### Startup Sequence
1. User selects Python venv in Setup tab
2. System checks/installs required packages
3. User clicks "Start Server"
4. Server starts via Electron IPC
5. Frontend polls `/status` until online
6. User can now upload data and run analysis

### Shutdown
- Click "Stop Server" in UI
- Or server auto-stops when Electron app closes
- `/shutdown` endpoint available for programmatic stop
