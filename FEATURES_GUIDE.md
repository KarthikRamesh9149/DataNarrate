# DataNarrate - Complete Features Guide
## Intent-Aware Data Science Copilot

---

## Executive Summary

DataNarrate is an AI-powered data analysis application that transforms raw datasets into actionable insights through automated cleaning, intelligent visualization, and machine learning. It uses natural language understanding to interpret user questions and tailors its entire analysis pipeline accordingly.

**Key Value Proposition**: Users describe what they want to learn in plain English, and DataNarrate automatically figures out what type of analysis to run, what to clean, what charts to generate, and what model to train.

---

## Core Features

### 1. Natural Language Intent Understanding

**What It Does**: Users type their analysis goal in plain English (e.g., "What factors determine match outcomes?" or "Who is the best salesperson?"), and the system intelligently interprets this to drive the entire analysis.

**How It Works**:
- LLM analyzes the user's question alongside the dataset's column names
- Automatically determines if this is:
  - **Exploratory Analysis (EDA)**: Pure data exploration, summaries, distributions
  - **Classification Task**: Predicting categories (win/lose, yes/no, product type)
  - **Regression Task**: Predicting numeric values (sales amount, price, score)
- Suggests the most appropriate target column for prediction tasks
- Provides confidence scores for each recommendation

**User Benefit**: No data science expertise required - just ask your question naturally.

---

### 2. Multi-Format Data Ingestion

**Supported Formats**:
- CSV files (with intelligent handling of embedded newlines and special characters)
- Excel files (.xlsx, .xls)
- JSON files

**Features**:
- Drag-and-drop or click-to-upload interface
- Built-in sample dataset (AFCON 2025-2026 football data) for testing
- Automatic preview of first 10 rows after upload
- Displays row count, column count, and column names immediately

**User Benefit**: Works with common data formats without any conversion needed.

---

### 3. Automated Data Profiling

**What It Analyzes**:
- **Column Types**: Automatically detects numeric, categorical, and datetime columns
- **Missing Values**: Counts and percentages per column
- **Unique Values**: Cardinality analysis for each column
- **Outliers**: IQR-based outlier detection for numeric columns
- **Duplicates**: Identifies exact duplicate rows
- **Data Quality Summary**: Plain-English summary of dataset health

**Statistics Provided**:
- Mean, median, standard deviation for numeric columns
- Top 5 most frequent values for categorical columns
- Min/max ranges for numeric columns
- Outlier percentages

**User Benefit**: Instant understanding of data quality without manual inspection.

---

### 4. Intelligent Data Cleaning

**Two-Phase Cleaning Approach**:

#### Phase 1: Safe Cleaning (Always Applied)
- **Column Name Normalization**: Converts to snake_case (e.g., "Team Name" becomes "team_name")
- **Whitespace Trimming**: Removes leading/trailing spaces from text values
- **Missing Value Standardization**: Converts N/A, null, None, -, etc. to proper NaN
- **Type Coercion**: Automatically converts strings to numbers or dates when appropriate
- **Duplicate Removal**: Removes exact duplicate rows

#### Phase 2: Recommended Cleaning (Optional)
- **Median Imputation**: Fills missing numeric values with column median
- **Mode Imputation**: Fills missing categorical values with most common value
- **Outlier Winsorization**: Clips extreme values to 1st-99th percentile
- **Rare Category Grouping**: Combines categories under 1% frequency into "Other"

**User Controls**:
- Review cleaning recommendations before applying
- See before/after comparison with exact counts
- Download cleaned dataset as CSV
- Full audit log of all cleaning actions

**User Benefit**: Production-ready data without manual cleaning scripts.

---

### 5. LLM-Driven Target Column Inference

**Intelligent Target Selection**:
- Analyzes user's question to understand what they want to predict
- Matches question intent to available columns
- Provides ranked list of target column candidates with confidence scores
- Flags potential data leakage risks (e.g., columns that encode the outcome)

**Smart Task Type Detection**:
- Examines actual data characteristics (not just LLM suggestion)
- Categorical targets or numeric with <=10 unique values → Classification
- Numeric targets with many unique values → Regression
- Prevents sklearn warnings about inappropriate task types

**User Controls**:
- View all recommended target columns
- See confidence percentage for each option
- Override automatic selection via dropdown
- Clear explanation of why each column was suggested

**User Benefit**: Ensures the right analysis type without data science knowledge.

---

### 6. Automated Visualization Generation

**Chart Types Generated**:

| Chart Type | Purpose | When Used |
|------------|---------|-----------|
| Missing Values Bar | Data quality overview | Always included |
| Numeric Distribution | Histogram showing value spread | For each numeric column |
| Categorical Frequency | Bar chart of category counts | For categorical columns |
| Correlation Heatmap | Relationships between numeric features | When 2+ numeric columns |
| Scatter Plot | Relationship between two variables | Pairs of numeric columns |
| Boxplot | Outlier detection and distribution comparison | Numeric columns |
| Time Trend | Values over time | When datetime column exists |
| Grouped Bar | Aggregated values by category | Category + numeric pairs |

**Intelligent Selection**:
- LLM analyzes user intent to prioritize relevant charts
- Always generates exactly 7 diverse visualizations
- Deduplication prevents redundant charts
- Column name translation handles renamed columns automatically

**Chart Features**:
- High-quality PNG output (100 DPI)
- Consistent styling across all charts
- Fallback to default plan if LLM unavailable
- Statistics embedded in each chart (correlation values, means, etc.)

**User Benefit**: Professional visualizations without matplotlib/seaborn expertise.

---

### 7. AI-Powered Chart Explanations

**Explanation Modes**:
- **Quick Mode**: 2-3 sentence summaries
- **Deep Mode**: 4-5 sentence detailed analysis

**Safety Features** (Prevents Misleading Claims):
- Never uses causal language ("causes", "drives", "leads to")
- Always uses correlation language ("associated with", "correlated with")
- Includes mandatory disclaimer: "This is exploratory data analysis, not causal inference"
- Special rules for correlation heatmaps to prevent "importance" claims
- Numeric grounding requirement - must cite specific values

**Relevance Scoring**:
Each chart explanation includes relevance rating (High/Medium/Low) based on user's original question.

**User Benefit**: Understand what charts mean without statistics background.

---

### 8. Machine Learning Model Training

**Supported Model Types**:
- **Classification**: Logistic Regression with automatic class balancing
- **Regression**: Ridge Regression with regularization

**Preprocessing Pipeline**:
- Automatic handling of missing values (median for numeric, mode for categorical)
- Standard scaling for numeric features
- One-hot encoding for categorical features
- Proper train/test split (default 80/20)

**Classification Metrics**:
- Accuracy (overall correct predictions)
- Macro F1 Score (balanced performance across classes)
- Confusion Matrix visualization

**Regression Metrics**:
- R-squared (variance explained)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Residuals plot visualization

**Feature Importance**:
- Top 20 features ranked by model coefficient magnitude
- Positive/negative effect indicators
- Automatic leakage detection for suspicious features

**User Controls**:
- Adjustable max iterations (100-5000)
- Configurable test size
- Random seed for reproducibility

**User Benefit**: Build predictive models without scikit-learn expertise.

---

### 9. Comprehensive Final Summary

**What It Provides**:
- Restates user's original question
- Explains analysis type performed (EDA vs. Model-based)
- Key findings with specific numbers
- Plain-English explanations of technical metrics
- Important caveats and limitations
- Clear bottom-line takeaway

**Context-Aware**:
- Different output for EDA-only vs. model-trained scenarios
- Includes feature importance discussion when model exists
- Warns about potential data leakage
- Notes dataset size limitations

**User Benefit**: Executive-ready summary without interpretation burden.

---

### 10. Export Capabilities

**Available Downloads**:
- **Cleaned Dataset**: CSV file with all cleaning applied
- **Charts ZIP**: All 7 visualizations in a single archive
- **Individual Charts**: Access any chart by filename

**User Benefit**: Take results into other tools or share with stakeholders.

---

## User Interface Tabs

### Setup Tab
- Python environment selection
- Dependency checking and installation
- Server start/stop controls
- CUDA availability indicator

### Data Tab
- Dataset preview (first 10 rows)
- Column headers
- Basic dataset info (rows x columns)

### Profile Tab
- Data quality summary
- Column-by-column statistics
- Missing value indicators
- Outlier warnings

### Cleaning Tab
- Before/after comparison
- Column rename mapping
- Cleaning action log
- Target column selector
- Apply cleaning button

### Charts Tab
- 7 visualizations in grid layout
- AI-generated explanations per chart
- Relevance ratings
- Download options

### Modeling Tab
- Target column confirmation
- Task type display
- Model configuration
- Performance metrics
- Confusion matrix / residuals plot
- Feature importance list
- Final summary panel

---

## Workflow Overview

```
1. SETUP
   └── Select Python venv → Install dependencies → Start server

2. LOAD DATA
   └── Upload file OR use sample dataset → See preview

3. DESCRIBE INTENT
   └── Type analysis goal in plain English

4. RUN ANALYSIS
   └── Click "Run Analysis" button
       ├── Profile data
       ├── Infer target column
       └── Show cleaning options

5. CLEAN DATA
   └── Review recommendations → Apply cleaning → See before/after

6. VIEW RESULTS
   ├── Charts generated automatically
   ├── Explanations generated per chart
   └── Navigate to Charts tab

7. BUILD MODEL (Optional)
   ├── Confirm target column
   ├── Click "Train Model"
   ├── View metrics and artifacts
   └── Read final summary

8. EXPORT
   └── Download cleaned CSV / charts ZIP
```

---

## Configuration Options

### Analysis Options
- **Allow Sample Rows to LLM**: Send up to 5 data rows for better context (privacy consideration)
- **Explanation Mode**: Quick (2-3 sentences) or Deep (4-5 sentences)

### Model Options
- **Max Iterations**: 100-5000 (default 1000)
- **Test Size**: Percentage of data for testing (default 20%)
- **Random Seed**: For reproducibility (default 42)

---

## Privacy & Safety Features

1. **No Raw Data in Summaries**: Final summaries use only statistics, not actual data values
2. **Leakage Detection**: Warns when features might encode target information
3. **Conservative Language**: AI never makes causal claims, only correlational
4. **Local Processing**: All analysis runs on user's machine, not cloud
5. **Optional Sample Rows**: User controls whether LLM sees actual data

---

## Technical Requirements

### Python Packages Required
- fastapi, uvicorn (web server)
- pandas, numpy (data processing)
- matplotlib (visualization)
- scikit-learn, scipy (machine learning)
- httpx (API calls)
- python-dotenv (configuration)
- openpyxl (Excel support)
- pydantic (data validation)
- joblib (model serialization)

### Groq API
- Requires Groq API key for LLM features
- Supports multiple API keys for capacity handling
- Model fallback: Primary → Fallback 1 → Fallback 2
- Works without LLM (uses rule-based fallbacks)

---

## Sample Use Cases

### Use Case 1: Sales Analysis
**Question**: "Who is my best salesperson and what makes them successful?"
- System detects regression task (predicting sales amount)
- Generates revenue distribution, salesperson comparison charts
- Trains model to identify factors correlated with high sales

### Use Case 2: Customer Churn
**Question**: "What factors predict customer churn?"
- System detects classification task (churn yes/no)
- Analyzes customer features
- Trains model, shows feature importance
- Identifies high-risk customer characteristics

### Use Case 3: Sports Analytics
**Question**: "What determines match outcomes?"
- System detects classification task (win/draw/lose)
- Generates team performance charts
- Trains model on match features
- Shows which statistics correlate with winning

### Use Case 4: Pure Exploration
**Question**: "Show me what's in this dataset"
- System detects EDA mode
- Generates comprehensive visualizations
- Provides data quality summary
- No model training (exploratory only)

---

## Version Information

- **Application**: DataNarrate v1.0
- **Backend**: FastAPI
- **Frontend**: React (dynamic window in ContextUI)
- **ML Framework**: scikit-learn
- **LLM Provider**: Groq API
