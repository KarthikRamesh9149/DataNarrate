"""
DataNarrate Server - Intent-Aware Data Science Copilot Backend
FastAPI server for data profiling, cleaning, visualization, and LLM-powered explanations.
"""

import os
import re
import json
import shutil
import zipfile
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - Multi-key support for capacity handling
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
]
# Filter out empty keys and fallback to legacy single key
GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k]
if not GROQ_API_KEYS:
    # Backward compatibility: use legacy GROQ_API_KEY if no numbered keys
    legacy_key = os.getenv("GROQ_API_KEY", "")
    if legacy_key:
        GROQ_API_KEYS = [legacy_key]

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Model routing - GLOBAL and IDENTICAL for all API keys
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "groq/compound")
FALLBACK_MODEL_1 = os.getenv("FALLBACK_MODEL_1", os.getenv("FALLBACK_MODEL", "llama-3.3-70b-versatile"))
FALLBACK_MODEL_2 = os.getenv("FALLBACK_MODEL_2", "")  # Optional third model

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8891"))

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CLEANED_DIR = OUTPUTS_DIR / "cleaned"
CHARTS_DIR = OUTPUTS_DIR / "charts"
MODELING_DIR = OUTPUTS_DIR / "modeling"
SAMPLE_DATASET_CSV = BASE_DIR / "afcon_2025_2026_dataset.csv"
SAMPLE_DATASET_XLSX = BASE_DIR / "afcon_2025_2026_dataset.xlsx"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
MODELING_DIR.mkdir(parents=True, exist_ok=True)

# App state
app_state: Dict[str, Any] = {
    "df": None,
    "df_cleaned": None,
    "dataset_name": None,
    "dataset_path": None,
    "profile": None,
    "cleaning_log": [],
    "viz_plan": [],
    "chart_stats": {},
    "column_name_mapping": {},
    "target_column": None,
    "target_candidates": [],
    "model": None,
    "model_results": None,
    "has_trained_model": False,
}

# FastAPI app
app = FastAPI(title="DataNarrate Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------
# Pydantic Models
# ---------------------

class StatusResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class IngestRequest(BaseModel):
    use_sample: bool = False
    dataset_path: Optional[str] = None


class ProfileRequest(BaseModel):
    pass


class PlanRequest(BaseModel):
    intent: str
    allow_sample_rows: bool = False


class CleanRequest(BaseModel):
    apply_recommended: bool = False


class ChartsRequest(BaseModel):
    intent: str


class ExplainRequest(BaseModel):
    mode: str = "quick"  # "quick" or "deep"
    intent: str = ""
    allow_sample_rows: bool = False


class TargetInferenceRequest(BaseModel):
    intent: str
    allow_sample_rows: bool = False


class SetTargetRequest(BaseModel):
    target_column: Optional[str] = None


class ModelTrainRequest(BaseModel):
    target_column: str
    task_type: str = "classification"  # "classification" | "regression" | "time_series"
    max_iter: int = 1000
    test_size: float = 0.2
    random_seed: int = 42


class ModelExplainRequest(BaseModel):
    mode: str = "quick"  # "quick" | "deep"
    intent: str = ""


# ---------------------
# Helper Functions
# ---------------------

def make_response(success: bool, data: Any = None, error: str = None) -> Dict:
    """Standard response format."""
    return {"success": success, "data": data, "error": error}


def to_snake_case(name: str) -> str:
    """Convert column name to snake_case."""
    # Replace spaces and special chars with underscores
    s = re.sub(r'[^\w\s]', '_', str(name))
    s = re.sub(r'\s+', '_', s)
    # Convert camelCase to snake_case
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    # Collapse multiple underscores and lowercase
    s = re.sub(r'_+', '_', s).strip('_').lower()
    return s if s else 'unnamed'


def normalize_for_matching(s: str) -> str:
    """Normalize a string for fuzzy column matching."""
    # Remove all non-alphanumeric, lowercase
    return re.sub(r'[^a-z0-9]', '', s.lower())


def word_overlap_score(s1: str, s2: str) -> float:
    """Calculate word overlap score between two strings."""
    # Split into words (by underscores, spaces, or camelCase)
    def get_words(s):
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)  # camelCase
        return set(re.split(r'[_\s]+', s.lower()))

    words1 = get_words(s1)
    words2 = get_words(s2)
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    return len(intersection) / max(len(words1), len(words2))


def translate_column_name(col_name: str, mapping: Dict[str, str], df_columns: List[str] = None) -> str:
    """Translate column name with aggressive fuzzy matching."""
    if not col_name:
        return col_name

    # 1. Exact match from mapping
    if mapping and col_name in mapping:
        return mapping[col_name]

    # 2. Exact match in df columns
    if df_columns and col_name in df_columns:
        return col_name

    # 3. Snake case version in df
    snake = to_snake_case(col_name)
    if df_columns and snake in df_columns:
        return snake

    # 4. Case-insensitive match with normalization
    if df_columns:
        col_lower = col_name.lower().replace(' ', '_').replace('-', '_')
        for df_col in df_columns:
            if df_col.lower() == col_lower:
                return df_col

    # 5. Try case-insensitive match from mapping values
    if mapping:
        col_lower = col_name.lower().strip()
        for orig, cleaned in mapping.items():
            if orig.lower().strip() == col_lower or cleaned.lower().strip() == col_lower:
                return cleaned

    # 6. Normalized string match (remove all non-alphanumeric)
    if df_columns:
        col_normalized = normalize_for_matching(col_name)
        for df_col in df_columns:
            df_normalized = normalize_for_matching(df_col)
            # Exact match after normalization
            if col_normalized == df_normalized:
                return df_col

    # 7. Partial/substring match (for LLM approximations)
    if df_columns:
        col_normalized = normalize_for_matching(col_name)
        for df_col in df_columns:
            df_normalized = normalize_for_matching(df_col)
            # Check if one contains the other (for abbreviations)
            if len(col_normalized) > 3 and len(df_normalized) > 3:
                if col_normalized in df_normalized or df_normalized in col_normalized:
                    return df_col

    # 8. Word overlap matching (for reordered or partial names)
    if df_columns:
        best_match = None
        best_score = 0.5  # Minimum threshold
        for df_col in df_columns:
            score = word_overlap_score(col_name, df_col)
            if score > best_score:
                best_score = score
                best_match = df_col
        if best_match:
            return best_match

    # 9. Character similarity (Jaccard-like)
    if df_columns:
        col_normalized = normalize_for_matching(col_name)
        best_match = None
        best_similarity = 0.7  # Higher threshold for character matching
        for df_col in df_columns:
            df_normalized = normalize_for_matching(df_col)
            if len(col_normalized) > 5 and len(df_normalized) > 5:
                # Character overlap
                col_chars = set(col_normalized)
                df_chars = set(df_normalized)
                intersection = len(col_chars & df_chars)
                union = len(col_chars | df_chars)
                similarity = intersection / union if union > 0 else 0
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = df_col
        if best_match:
            return best_match

    # 10. Last resort: return snake_case version
    return snake if mapping else col_name


def find_best_column_match(requested: str, available: List[str], col_type: str = None) -> Optional[str]:
    """Find the best matching column from available columns, with type preference."""
    if not requested or not available:
        return None

    # First try direct translation
    matched = translate_column_name(requested, {}, available)
    if matched in available:
        return matched

    # If still not found and type hint given, find any column of that type
    return None


def translate_chart_config(config: Dict, mapping: Dict[str, str], df_columns: List[str] = None) -> Dict:
    """Translate all column references in chart config from original to cleaned names."""
    if not config:
        return config
    translated = config.copy()
    # Translate single column references
    for key in ['column', 'x_column', 'y_column', 'date_column', 'value_column', 'group_column']:
        if key in translated and translated[key]:
            translated[key] = translate_column_name(translated[key], mapping, df_columns)
    # Translate list of columns
    if 'columns' in translated and isinstance(translated['columns'], list):
        translated['columns'] = [translate_column_name(c, mapping, df_columns) for c in translated['columns']]
    return translated


def normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common missing value tokens with NaN."""
    missing_tokens = ['', 'N/A', 'n/a', 'NA', 'na', 'NULL', 'null', 'None', 'none',
                      'NaN', 'nan', '-', '--', '.', '?', 'missing', 'MISSING']
    return df.replace(missing_tokens, np.nan)


def compute_outlier_pct(series: pd.Series) -> float:
    """Compute percentage of outliers using IQR method."""
    if series.dtype not in ['int64', 'float64'] or series.isna().all():
        return 0.0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((series < lower) | (series > upper)).sum()
    return round(100 * outliers / len(series), 2) if len(series) > 0 else 0.0


def detect_datetime_column(series: pd.Series) -> bool:
    """Check if a column looks like a datetime."""
    if series.dtype == 'datetime64[ns]':
        return True
    if series.dtype == 'object':
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return False
        try:
            pd.to_datetime(sample)
            return True
        except:
            return False
    return False


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data profile with robust error handling."""
    try:
        n_rows, n_cols = df.shape

        column_profiles = []
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []

        for col in df.columns:
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_pct = round(100 * missing_count / n_rows, 2) if n_rows > 0 else 0

            col_profile = {
                "name": col,
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": int(series.nunique()),
                "is_constant": series.nunique() <= 1,
            }

            # Check if datetime
            if detect_datetime_column(series):
                datetime_cols.append(col)
                col_profile["column_type"] = "datetime"
            # Numeric column
            elif pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                col_profile["column_type"] = "numeric"
                valid = series.dropna()
                if len(valid) > 0:
                    col_profile["min"] = float(valid.min())
                    col_profile["max"] = float(valid.max())
                    col_profile["mean"] = round(float(valid.mean()), 4)
                    col_profile["std"] = round(float(valid.std()), 4)
                    col_profile["outlier_pct"] = compute_outlier_pct(valid)
                else:
                    col_profile["min"] = col_profile["max"] = col_profile["mean"] = col_profile["std"] = None
                    col_profile["outlier_pct"] = 0
            # Categorical column
            else:
                categorical_cols.append(col)
                col_profile["column_type"] = "categorical"
                col_profile["high_cardinality"] = col_profile["unique_count"] > 50
                # Top values
                top = series.value_counts().head(5)
                col_profile["top_values"] = {str(k): int(v) for k, v in top.items()}

            column_profiles.append(col_profile)

        # Duplicate detection
        duplicates_count = int(df.duplicated().sum())

        # Summary text
        high_missing = [p for p in column_profiles if p["missing_pct"] > 10]
        constant_cols = [p for p in column_profiles if p["is_constant"]]

        summary_parts = [
            f"Dataset has {n_rows} rows and {n_cols} columns.",
            f"Numeric columns: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, Datetime: {len(datetime_cols)}.",
        ]

        if duplicates_count > 0:
            summary_parts.append(f"Found {duplicates_count} duplicate rows.")

        if high_missing:
            cols = ", ".join([p["name"] for p in high_missing[:3]])
            summary_parts.append(f"Columns with >10% missing: {cols}{'...' if len(high_missing) > 3 else ''}.")

        if constant_cols:
            summary_parts.append(f"{len(constant_cols)} constant/near-constant columns detected.")

        profile = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "duplicates_count": duplicates_count,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "column_profiles": column_profiles,
            "summary_text": " ".join(summary_parts),
        }

        return profile

    except Exception as e:
        # Return minimal profile on error to ensure robustness
        print(f"Error in profile_dataframe: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "n_rows": len(df) if df is not None else 0,
            "n_cols": len(df.columns) if df is not None else 0,
            "duplicates_count": 0,
            "numeric_cols": list(df.select_dtypes(include=[np.number]).columns) if df is not None else [],
            "categorical_cols": list(df.select_dtypes(include=['object', 'category']).columns) if df is not None else [],
            "datetime_cols": [],
            "column_profiles": [],
            "summary_text": f"Dataset with {len(df) if df is not None else 0} rows and {len(df.columns) if df is not None else 0} columns (profiling encountered an error)"
        }


def apply_safe_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, List[Dict]]:
    """Apply safe cleaning operations (automatic)."""
    log = []
    df = df.copy()
    original_rows = len(df)

    # 1. Normalize column names to snake_case
    old_cols = list(df.columns)
    name_mapping = {}
    new_cols = []
    for col in old_cols:
        new_name = to_snake_case(col)
        # Handle duplicates
        base = new_name
        counter = 1
        while new_name in new_cols:
            new_name = f"{base}_{counter}"
            counter += 1
        new_cols.append(new_name)
        if col != new_name:
            name_mapping[col] = new_name

    df.columns = new_cols
    app_state["column_name_mapping"] = name_mapping

    if name_mapping:
        log.append({
            "action": "normalize_column_names",
            "details": f"Renamed {len(name_mapping)} columns to snake_case",
            "mapping": name_mapping
        })

    # 2. Trim string columns
    str_cols = df.select_dtypes(include=['object']).columns
    trimmed_count = 0
    for col in str_cols:
        before = df[col].copy()
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        if not before.equals(df[col]):
            trimmed_count += 1

    if trimmed_count > 0:
        log.append({
            "action": "trim_whitespace",
            "details": f"Trimmed whitespace in {trimmed_count} string columns"
        })

    # 3. Normalize missing tokens
    df_before_missing = df.isna().sum().sum()
    df = normalize_missing_tokens(df)
    df_after_missing = df.isna().sum().sum()
    new_nulls = df_after_missing - df_before_missing

    if new_nulls > 0:
        log.append({
            "action": "normalize_missing_tokens",
            "details": f"Converted {new_nulls} missing value tokens to NaN"
        })

    # 4. Type coercion attempts
    coerced = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try numeric
            try:
                numeric = pd.to_numeric(df[col], errors='coerce')
                non_null_orig = df[col].notna().sum()
                non_null_new = numeric.notna().sum()
                if non_null_new >= non_null_orig * 0.8 and non_null_new > 0:
                    df[col] = numeric
                    coerced.append(f"{col} -> numeric")
                    continue
            except:
                pass

            # Try datetime
            try:
                dt = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                non_null_orig = df[col].notna().sum()
                non_null_new = dt.notna().sum()
                if non_null_new >= non_null_orig * 0.8 and non_null_new > 0:
                    df[col] = dt
                    coerced.append(f"{col} -> datetime")
            except:
                pass

    if coerced:
        log.append({
            "action": "type_coercion",
            "details": f"Coerced {len(coerced)} columns: {', '.join(coerced[:5])}{'...' if len(coerced) > 5 else ''}"
        })

    # 5. Remove exact duplicates
    df_deduped = df.drop_duplicates()
    removed = original_rows - len(df_deduped)

    if removed > 0:
        df = df_deduped
        log.append({
            "action": "remove_duplicates",
            "details": f"Removed {removed} duplicate rows"
        })

    # Summary
    log.append({
        "action": "safe_cleaning_complete",
        "details": f"Safe cleaning finished. Rows: {original_rows} -> {len(df)}",
        "rows_before": original_rows,
        "rows_after": len(df)
    })

    return df, log


def apply_recommended_cleaning(df: pd.DataFrame, profile: Dict) -> tuple[pd.DataFrame, List[Dict]]:
    """Apply recommended cleaning operations (user-triggered)."""
    log = []
    df = df.copy()

    # 1. Median imputation for numeric columns
    numeric_cols = [p["name"] for p in profile["column_profiles"]
                    if p.get("column_type") == "numeric" and p["missing_count"] > 0]

    imputed_numeric = []
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
                imputed_numeric.append(col)

    if imputed_numeric:
        log.append({
            "action": "median_imputation",
            "details": f"Imputed {len(imputed_numeric)} numeric columns with median: {', '.join(imputed_numeric[:5])}"
        })

    # 2. Mode imputation for categorical columns
    cat_cols = [p["name"] for p in profile["column_profiles"]
                if p.get("column_type") == "categorical" and p["missing_count"] > 0]

    imputed_cat = []
    for col in cat_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
                imputed_cat.append(col)

    if imputed_cat:
        log.append({
            "action": "mode_imputation",
            "details": f"Imputed {len(imputed_cat)} categorical columns with mode: {', '.join(imputed_cat[:5])}"
        })

    # 3. Winsorize numeric outliers (1st-99th percentile)
    winsorized = []
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col]
        if series.isna().all():
            continue
        p1 = series.quantile(0.01)
        p99 = series.quantile(0.99)
        outliers_before = ((series < p1) | (series > p99)).sum()
        if outliers_before > 0:
            df[col] = series.clip(lower=p1, upper=p99)
            winsorized.append(col)

    if winsorized:
        log.append({
            "action": "winsorize_outliers",
            "details": f"Winsorized outliers in {len(winsorized)} columns (1st-99th percentile)"
        })

    # 4. Group rare categories (<1%) -> "Other"
    grouped = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        vc = df[col].value_counts(normalize=True)
        rare = vc[vc < 0.01].index.tolist()
        if len(rare) > 0:
            df[col] = df[col].replace(rare, "Other")
            grouped.append(col)

    if grouped:
        log.append({
            "action": "group_rare_categories",
            "details": f"Grouped rare categories (<1%) to 'Other' in {len(grouped)} columns"
        })

    log.append({
        "action": "recommended_cleaning_complete",
        "details": f"Recommended cleaning finished. Final rows: {len(df)}"
    })

    return df, log


async def call_llm(messages: List[Dict], json_mode: bool = True) -> Optional[Dict]:
    """
    Call Groq LLM with model fallback and multi-key capacity handling.

    Model routing: PRIMARY_MODEL -> FALLBACK_MODEL_1 -> FALLBACK_MODEL_2 (fixed order)
    Key rotation: On quota/429 errors, automatically try next API key (invisible to user)
    """
    if not GROQ_API_KEYS:
        return None

    # Build model list - remove 'groq/' prefix if present
    models = [PRIMARY_MODEL.replace("groq/", "")]
    if FALLBACK_MODEL_1:
        models.append(FALLBACK_MODEL_1.replace("groq/", ""))
    if FALLBACK_MODEL_2:
        models.append(FALLBACK_MODEL_2.replace("groq/", ""))

    # Try each model in order (model fallback for reasoning errors)
    for model in models:
        # Try each API key in order (key rotation for quota/capacity errors)
        for key_idx, api_key in enumerate(GROQ_API_KEYS):
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2000,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        f"{GROQ_BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload
                    )

                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        if json_mode:
                            return json.loads(content)
                        return {"text": content}

                    # Check for quota/rate-limit errors - rotate to next key
                    elif response.status_code == 429 or "quota" in response.text.lower() or "rate" in response.text.lower():
                        print(f"LLM quota/rate-limit ({model}, key #{key_idx+1}): {response.status_code}")
                        continue  # Try next API key

                    else:
                        print(f"LLM error ({model}, key #{key_idx+1}): {response.status_code} - {response.text[:200]}")
                        break  # Non-quota error, try next model

            except httpx.TimeoutException:
                print(f"LLM timeout ({model}, key #{key_idx+1})")
                continue  # Try next key on timeout

            except Exception as e:
                print(f"LLM exception ({model}, key #{key_idx+1}): {e}")
                continue  # Try next key on exception

    return None


def validate_and_fix_columns(chart_type: str, config: Dict, df: pd.DataFrame, profile: Dict) -> Dict:
    """Validate column references and substitute with valid alternatives if needed."""
    fixed_config = config.copy()
    df_columns = list(df.columns)
    numeric_cols = profile.get("numeric_cols", [])
    categorical_cols = profile.get("categorical_cols", [])
    datetime_cols = profile.get("datetime_cols", [])

    def get_valid_column(col_name: str, preferred_type: str = None) -> Optional[str]:
        """Get a valid column, with fallback to any column of preferred type."""
        if not col_name:
            return None

        # Try exact match
        if col_name in df_columns:
            return col_name

        # Try translation
        translated = translate_column_name(col_name, {}, df_columns)
        if translated in df_columns:
            return translated

        # Fallback: return first available column of preferred type
        if preferred_type == "numeric" and numeric_cols:
            return numeric_cols[0]
        elif preferred_type == "categorical" and categorical_cols:
            return categorical_cols[0]
        elif preferred_type == "datetime" and datetime_cols:
            return datetime_cols[0]

        return None

    # Fix columns based on chart type
    if chart_type == "scatter":
        x_col = get_valid_column(config.get("x_column"), "numeric")
        y_col = get_valid_column(config.get("y_column"), "numeric")

        # Ensure x and y are different
        if x_col and y_col and x_col == y_col and len(numeric_cols) >= 2:
            y_col = [c for c in numeric_cols if c != x_col][0]

        # If we couldn't find valid columns, use first two numeric cols
        if not x_col or not y_col:
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]

        fixed_config["x_column"] = x_col
        fixed_config["y_column"] = y_col

    elif chart_type == "grouped_bar":
        group_col = get_valid_column(config.get("group_column"), "categorical")
        value_col = get_valid_column(config.get("value_column"), "numeric")

        # Fallback to first available
        if not group_col and categorical_cols:
            group_col = categorical_cols[0]
        if not value_col and numeric_cols:
            value_col = numeric_cols[0]

        fixed_config["group_column"] = group_col
        fixed_config["value_column"] = value_col

    elif chart_type == "numeric_distribution":
        col = get_valid_column(config.get("column"), "numeric")
        if not col and numeric_cols:
            col = numeric_cols[0]
        fixed_config["column"] = col

    elif chart_type == "categorical_frequency":
        col = get_valid_column(config.get("column"), "categorical")
        if not col and categorical_cols:
            col = categorical_cols[0]
        fixed_config["column"] = col

    elif chart_type == "time_trend":
        date_col = get_valid_column(config.get("date_column"), "datetime")
        value_col = get_valid_column(config.get("value_column"), "numeric")

        if not date_col and datetime_cols:
            date_col = datetime_cols[0]
        if not value_col and numeric_cols:
            value_col = numeric_cols[0]

        fixed_config["date_column"] = date_col
        fixed_config["value_column"] = value_col

    elif chart_type == "boxplot_outliers":
        cols = config.get("columns", [])
        valid_cols = []
        for c in cols:
            valid = get_valid_column(c, "numeric")
            if valid and valid not in valid_cols:
                valid_cols.append(valid)
        # If no valid columns, use first few numeric cols
        if not valid_cols:
            valid_cols = numeric_cols[:6]
        fixed_config["columns"] = valid_cols

    return fixed_config


def generate_chart(chart_type: str, df: pd.DataFrame, profile: Dict,
                   chart_config: Dict, output_path: Path) -> Dict[str, Any]:
    """Generate a single chart and return stats."""
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Validate and fix column references before generating chart
    chart_config = validate_and_fix_columns(chart_type, chart_config, df, profile)

    stats = {"type": chart_type, "config": chart_config}

    try:
        if chart_type == "missing_values_bar":
            # Missing values bar chart
            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=True)
            if len(missing) == 0:
                missing = pd.Series([0], index=["No missing values"])

            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(missing)))
            plt.barh(range(len(missing)), missing.values, color=colors)
            plt.yticks(range(len(missing)), missing.index)
            plt.xlabel("Missing Count")
            plt.title("Missing Values by Column")
            stats["total_missing"] = int(missing.sum())
            stats["cols_with_missing"] = len(missing)

        elif chart_type == "numeric_distribution":
            col = chart_config.get("column")
            if col and col in df.columns:
                data = df[col].dropna()
                plt.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.title(f"Distribution of {col}")
                stats["mean"] = round(float(data.mean()), 4)
                stats["median"] = round(float(data.median()), 4)
                stats["std"] = round(float(data.std()), 4)
                stats["skewness"] = round(float(data.skew()), 4)
            else:
                plt.text(0.5, 0.5, "Column not found", ha='center', va='center')

        elif chart_type == "categorical_frequency":
            col = chart_config.get("column")
            top_n = chart_config.get("top_n", 10)
            if col and col in df.columns:
                vc = df[col].value_counts().head(top_n)
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(vc)))
                plt.barh(range(len(vc)), vc.values, color=colors)
                plt.yticks(range(len(vc)), [str(x)[:30] for x in vc.index])
                plt.xlabel("Count")
                plt.title(f"Top {len(vc)} Categories in {col}")
                stats["top_category"] = str(vc.index[0]) if len(vc) > 0 else None
                stats["top_count"] = int(vc.values[0]) if len(vc) > 0 else 0
                stats["unique_shown"] = len(vc)
            else:
                plt.text(0.5, 0.5, "Column not found", ha='center', va='center')

        elif chart_type == "correlation_heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr()
                im = plt.matshow(corr, cmap='RdBu_r', fignum=False, vmin=-1, vmax=1)
                plt.colorbar(im)
                plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='left')
                plt.yticks(range(len(corr.columns)), corr.columns)
                plt.title("Correlation Heatmap", pad=20)
                # Find strongest correlations (excluding diagonal)
                corr_vals = corr.values.copy()
                np.fill_diagonal(corr_vals, 0)
                max_corr = np.abs(corr_vals).max()
                stats["max_correlation"] = round(float(max_corr), 4)
                stats["num_features"] = len(corr.columns)
            else:
                plt.text(0.5, 0.5, "Not enough numeric columns", ha='center', va='center')

        elif chart_type == "scatter":
            x_col = chart_config.get("x_column")
            y_col = chart_config.get("y_column")
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                valid = df[[x_col, y_col]].dropna()
                plt.scatter(valid[x_col], valid[y_col], alpha=0.6, c='steelblue', edgecolor='white')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"{y_col} vs {x_col}")
                if len(valid) > 1:
                    corr = valid[x_col].corr(valid[y_col])
                    stats["correlation"] = round(float(corr), 4) if pd.notna(corr) else None
                stats["n_points"] = len(valid)
            else:
                plt.text(0.5, 0.5, "Columns not found", ha='center', va='center')

        elif chart_type == "boxplot_outliers":
            cols = chart_config.get("columns", [])
            if not cols:
                cols = list(df.select_dtypes(include=[np.number]).columns)[:6]
            if cols:
                data = [df[c].dropna() for c in cols if c in df.columns]
                labels = [c for c in cols if c in df.columns]
                if data:
                    plt.boxplot(data, labels=labels)
                    plt.xticks(rotation=45, ha='right')
                    plt.title("Boxplot - Outlier Detection")
                    stats["columns"] = labels
                else:
                    plt.text(0.5, 0.5, "No valid columns", ha='center', va='center')
            else:
                plt.text(0.5, 0.5, "No numeric columns", ha='center', va='center')

        elif chart_type == "time_trend":
            date_col = chart_config.get("date_column")
            value_col = chart_config.get("value_column")
            if date_col and value_col and date_col in df.columns and value_col in df.columns:
                temp = df[[date_col, value_col]].dropna().copy()
                if temp[date_col].dtype != 'datetime64[ns]':
                    temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
                temp = temp.dropna().sort_values(date_col)
                plt.plot(temp[date_col], temp[value_col], marker='o', markersize=4, alpha=0.7)
                plt.xlabel(date_col)
                plt.ylabel(value_col)
                plt.title(f"{value_col} Over Time")
                plt.xticks(rotation=45)
                stats["n_points"] = len(temp)
                stats["date_range"] = f"{temp[date_col].min()} to {temp[date_col].max()}"
            else:
                plt.text(0.5, 0.5, "Date/value columns not found", ha='center', va='center')

        elif chart_type == "grouped_bar":
            group_col = chart_config.get("group_column")
            value_col = chart_config.get("value_column")
            agg = chart_config.get("aggregation", "mean")
            if group_col and value_col and group_col in df.columns and value_col in df.columns:
                grouped = df.groupby(group_col)[value_col].agg(agg).sort_values(ascending=False).head(10)
                colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(grouped)))
                plt.barh(range(len(grouped)), grouped.values, color=colors)
                plt.yticks(range(len(grouped)), [str(x)[:25] for x in grouped.index])
                plt.xlabel(f"{agg.title()} of {value_col}")
                plt.title(f"{value_col} by {group_col}")
                stats["top_group"] = str(grouped.index[0]) if len(grouped) > 0 else None
                stats["top_value"] = round(float(grouped.values[0]), 2) if len(grouped) > 0 else None
            else:
                plt.text(0.5, 0.5, "Columns not found", ha='center', va='center')

        else:
            plt.text(0.5, 0.5, f"Unknown chart type: {chart_type}", ha='center', va='center')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

        stats["success"] = True
        stats["path"] = str(output_path)

    except Exception as e:
        plt.close()
        stats["success"] = False
        stats["error"] = str(e)
        # Create placeholder
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)[:100]}", ha='center', va='center', wrap=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    return stats


def generate_default_viz_plan(profile: Dict) -> List[Dict]:
    """Generate a default visualization plan based on data profile."""
    plan = []
    numeric_cols = profile.get("numeric_cols", [])
    categorical_cols = profile.get("categorical_cols", [])
    datetime_cols = profile.get("datetime_cols", [])

    # 1. Always start with missing values
    plan.append({"type": "missing_values_bar", "config": {}})

    # 2. Numeric distribution (first numeric col)
    if numeric_cols:
        plan.append({"type": "numeric_distribution", "config": {"column": numeric_cols[0]}})

    # 3. Categorical frequency (first categorical col)
    if categorical_cols:
        plan.append({"type": "categorical_frequency", "config": {"column": categorical_cols[0], "top_n": 10}})

    # 4. Correlation heatmap (if multiple numeric)
    if len(numeric_cols) >= 2:
        plan.append({"type": "correlation_heatmap", "config": {}})

    # 5. Scatter plot (first two numeric)
    if len(numeric_cols) >= 2:
        plan.append({"type": "scatter", "config": {"x_column": numeric_cols[0], "y_column": numeric_cols[1]}})

    # 6. Boxplot
    if numeric_cols:
        plan.append({"type": "boxplot_outliers", "config": {"columns": numeric_cols[:6]}})

    # 7. Time trend OR grouped bar
    if datetime_cols and numeric_cols:
        plan.append({"type": "time_trend", "config": {"date_column": datetime_cols[0], "value_column": numeric_cols[0]}})
    elif categorical_cols and numeric_cols:
        plan.append({"type": "grouped_bar", "config": {"group_column": categorical_cols[0], "value_column": numeric_cols[0], "aggregation": "mean"}})

    # Ensure exactly 7 charts
    while len(plan) < 7:
        # Add more numeric distributions or categorical frequencies
        idx = len(plan) - 1
        if idx < len(numeric_cols):
            plan.append({"type": "numeric_distribution", "config": {"column": numeric_cols[min(idx, len(numeric_cols)-1)]}})
        elif idx - len(numeric_cols) < len(categorical_cols):
            cidx = idx - len(numeric_cols)
            plan.append({"type": "categorical_frequency", "config": {"column": categorical_cols[min(cidx, len(categorical_cols)-1)], "top_n": 10}})
        else:
            plan.append({"type": "missing_values_bar", "config": {}})

    return plan[:7]


# ---------------------
# Modeling Helper Functions
# ---------------------

def build_preprocessing_pipeline(numeric_cols: List[str], categorical_cols: List[str]):
    """Build sklearn ColumnTransformer for preprocessing."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    return preprocessor


def generate_confusion_matrix_plot(y_true, y_pred, labels, output_path: Path):
    """Generate confusion matrix visualization."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    # Use matplotlib for heatmap (avoid seaborn dependency)
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, facecolor='white')
    plt.close()


def generate_residuals_plot(y_true, y_pred, output_path: Path):
    """Generate residuals plot for regression."""
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='steelblue', edgecolor='white')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, facecolor='white')
    plt.close()


# ---------------------
# API Endpoints
# ---------------------

@app.get("/status")
async def get_status():
    """Health check endpoint."""
    return make_response(True, {
        "status": "online",
        "dataset_loaded": app_state["df"] is not None,
        "dataset_name": app_state["dataset_name"],
    })


@app.post("/ingest")
async def ingest_dataset(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False)
):
    """Load a dataset from file upload or sample."""
    try:
        df = None
        dataset_name = None

        if use_sample:
            # Load sample dataset
            if SAMPLE_DATASET_XLSX.exists():
                df = pd.read_excel(SAMPLE_DATASET_XLSX, engine='openpyxl')
                dataset_name = "AFCON 2025-2026 (Sample)"
                app_state["dataset_path"] = str(SAMPLE_DATASET_XLSX)
            elif SAMPLE_DATASET_CSV.exists():
                # Use python engine for better handling of quoted fields with embedded newlines
                df = pd.read_csv(SAMPLE_DATASET_CSV, engine='python', on_bad_lines='warn')
                # Clean up any embedded newlines in string columns
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].apply(lambda x: x.replace('\n', ' ').strip() if isinstance(x, str) else x)
                dataset_name = "AFCON 2025-2026 (Sample)"
                app_state["dataset_path"] = str(SAMPLE_DATASET_CSV)
            else:
                return make_response(False, error="Sample dataset not found")

        elif file:
            # Save uploaded file
            filename = file.filename or "uploaded_dataset"
            ext = Path(filename).suffix.lower()
            save_path = DATA_DIR / filename

            with open(save_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Load based on extension
            if ext == ".csv":
                # Use python engine for better handling of quoted fields with embedded newlines
                df = pd.read_csv(save_path, engine='python', on_bad_lines='warn')
                # Clean up any embedded newlines in string columns
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].apply(lambda x: x.replace('\n', ' ').strip() if isinstance(x, str) else x)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(save_path, engine='openpyxl')
            elif ext == ".json":
                df = pd.read_json(save_path)
            else:
                return make_response(False, error=f"Unsupported file format: {ext}")

            dataset_name = filename
            app_state["dataset_path"] = str(save_path)
        else:
            return make_response(False, error="No dataset provided")

        # Reset ALL app state for fresh analysis (important when switching datasets)
        app_state["df"] = df
        app_state["df_cleaned"] = None
        app_state["dataset_name"] = dataset_name
        app_state["dataset_path"] = app_state.get("dataset_path")  # Keep path set above
        app_state["profile"] = None
        app_state["cleaning_log"] = []
        app_state["viz_plan"] = []
        app_state["chart_stats"] = {}
        app_state["column_name_mapping"] = {}
        # Reset target and modeling state
        app_state["target_column"] = None
        app_state["target_candidates"] = []
        app_state["model"] = None
        app_state["model_results"] = None
        app_state["has_trained_model"] = False

        # Return preview
        preview = df.head(10).fillna("").to_dict(orient="records")
        columns = list(df.columns)

        return make_response(True, {
            "dataset_name": dataset_name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": columns,
            "preview": preview,
        })

    except Exception as e:
        return make_response(False, error=str(e))


@app.post("/profile")
async def profile_dataset():
    """Generate data quality profile. Uses cleaned DataFrame if available."""
    try:
        # Use cleaned DataFrame if available, otherwise original
        df = app_state.get("df_cleaned")
        if df is None:
            df = app_state.get("df")
        if df is None:
            return make_response(False, error="No dataset loaded")

        profile = profile_dataframe(df)
        app_state["profile"] = profile

        return make_response(True, profile)

    except Exception as e:
        return make_response(False, error=str(e))


def is_id_like_column(col_name: str, unique_count: int, total_rows: int) -> bool:
    """Check if column appears to be an ID column."""
    id_patterns = ['id', '_id', 'uuid', 'index', 'pk', 'key']
    name_lower = col_name.lower()
    # Name-based check
    if any(p in name_lower for p in id_patterns):
        return True
    # Uniqueness check: if unique values are near row count, likely ID
    if total_rows > 10 and unique_count >= total_rows * 0.95:
        return True
    return False


def is_leakage_column(col_name: str) -> bool:
    """Check if column name suggests post-outcome leakage."""
    leakage_patterns = ['score', 'result', 'winner', 'loser', 'outcome', 'final',
                        'points', 'goals', 'target', 'label', 'actual']
    name_lower = col_name.lower()
    return any(p in name_lower for p in leakage_patterns)


@app.post("/infer-target")
async def infer_target_column(request: TargetInferenceRequest):
    """
    Use LLM to intelligently analyze user intent + column names to determine:
    1. Is this a prediction task or exploratory analysis?
    2. If prediction, what is the best target column?

    This is fully LLM-driven - no hardcoded keyword matching.
    """
    try:
        df = app_state.get("df_cleaned")
        if df is None:
            df = app_state.get("df")
        if df is None:
            return make_response(False, error="No dataset loaded")

        profile = app_state.get("profile")
        if profile is None:
            profile = profile_dataframe(df)
            app_state["profile"] = profile

        # Build schema summary for LLM - this is what the LLM needs to make smart decisions
        schema_summary = []
        for col_profile in profile.get("column_profiles", []):
            col_info = {
                "name": col_profile["name"],
                "type": col_profile.get("column_type", "unknown"),
                "unique_count": col_profile.get("unique_count", 0),
                "missing_pct": col_profile.get("missing_pct", 0),
                "is_constant": col_profile.get("is_constant", False),
            }
            # Add type-specific info
            if col_profile.get("column_type") == "categorical":
                col_info["high_cardinality"] = col_profile.get("high_cardinality", False)
            schema_summary.append(col_info)

        # Build context for LLM
        context = {
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
            "schema": schema_summary,
            "numeric_cols": profile.get("numeric_cols", []),
            "categorical_cols": profile.get("categorical_cols", []),
        }

        # Optionally add sample rows
        if request.allow_sample_rows:
            context["sample_rows"] = df.head(5).fillna("").to_dict(orient="records")

        system_prompt = """You are a smart data science assistant. Your job is to analyze the user's question AND the dataset columns together to determine what kind of analysis is needed.

STEP 1: Understand the user's question
- What are they trying to learn, predict, or understand?
- Are they asking about outcomes, rankings, comparisons, or just exploration?

STEP 2: Match question to columns
- Look at the column names - do any directly relate to what they're asking?
- Example: "Who is the best salesperson?" + column "Salesperson" = that's likely what they want to analyze
- Example: "What determines match results?" + column "FullTimeResult" = that's likely the target

STEP 3: Decide task type based on the TARGET COLUMN's characteristics
- "eda" = Pure exploration, summaries, distributions (e.g., "show me the data", "what's in this dataset?")
- "classification" = When target is CATEGORICAL or has FEW unique values (<=10), like win/lose, yes/no, categories
- "regression" = When target is NUMERIC with MANY unique values (like amounts, prices, scores, percentages)

HOW TO CHOOSE CLASSIFICATION VS REGRESSION:
- Look at the target column's TYPE and UNIQUE COUNT
- If target is categorical (text categories) -> classification
- If target is numeric with <=10 unique values -> classification (it's like categories coded as numbers)
- If target is numeric with many unique values (>10) -> regression
- Example: "Amount" with range 100.0 to 50000.0 -> regression (continuous numbers)
- Example: "Result" with 3 unique values ['Win', 'Lose', 'Draw'] -> classification

IMPORTANT RULES:
- If the question mentions finding "best", "top", "highest", "winner" etc., look for a NUMERIC column that measures performance (amounts, scores, counts)
- If asking about "what affects/influences/determines X", X is likely the target
- Only return "eda" if the question is truly just exploration with no prediction goal
- Do NOT make the user's name column (like "Salesperson", "Player", "Team") the target - find the OUTCOME column they should predict
- For "who is the best X?", the target should be the NUMERIC metric that defines "best" (sales amount, wins count, score, etc.) - this is usually REGRESSION

Return STRICT JSON:
{
  "task_type": "classification|regression|eda",
  "target_candidates": [
    {"column": "col_name", "confidence": 0.0-1.0, "reason": "short explanation"}
  ],
  "chosen_target": "best_column_name" or null,
  "needs_user_confirmation": true,
  "assumption_note": "one sentence explaining what we're predicting and why"
}

CRITICAL RULES FOR target_candidates:
- ALWAYS return AT LEAST 3 candidate columns (unless there are fewer than 3 suitable columns)
- Order them by confidence (highest first)
- Include numeric columns that could be targets AND categorical columns with few unique values
- Give the user OPTIONS to choose from - don't just return 1 column
- Only return empty target_candidates if task_type is "eda"."""

        # Build a detailed column list for the prompt - include sample values for categorical columns
        column_list = []
        for col in schema_summary:
            col_info = f"- {col['name']} ({col['type']}, {col['unique_count']} unique values)"

            # Add sample values for categorical columns to help LLM understand the data
            if col['type'] == 'categorical' and col['unique_count'] <= 10:
                try:
                    sample_vals = df[col['name']].dropna().unique()[:5].tolist()
                    if sample_vals:
                        col_info += f" -> examples: {sample_vals}"
                except:
                    pass
            elif col['type'] == 'numeric':
                try:
                    col_data = df[col['name']].dropna()
                    if len(col_data) > 0:
                        col_info += f" -> range: {col_data.min():.1f} to {col_data.max():.1f}"
                except:
                    pass

            column_list.append(col_info)

        user_prompt = f"""User's question: "{request.intent}"

Dataset has {profile['n_rows']} rows and these columns:
{chr(10).join(column_list)}

Analyze the question and columns together:
1. What is the user trying to learn or predict?
2. Is this a prediction task (needs a target column) or just exploration (EDA)?
3. If prediction, which column should be the target?

Return your analysis as JSON."""

        llm_response = await call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        if llm_response:
            # Validate LLM response with deterministic checks
            candidates = llm_response.get("target_candidates", [])
            validated_candidates = []

            for candidate in candidates:
                col_name = candidate.get("column", "")
                if col_name not in df.columns:
                    continue  # Skip non-existent columns

                # Get column profile
                col_profile = next((p for p in profile["column_profiles"] if p["name"] == col_name), None)
                if not col_profile:
                    continue

                unique_count = col_profile.get("unique_count", 0)

                # Reject ID-like columns
                if is_id_like_column(col_name, unique_count, len(df)):
                    continue

                # Flag potential leakage columns (but don't reject - user may want them)
                is_leakage = is_leakage_column(col_name)

                # Accept the column - model training will auto-detect the right task type
                col_type = col_profile.get("column_type", "unknown")

                validated_candidates.append({
                    **candidate,
                    "is_leakage_risk": is_leakage,
                    "col_type": col_type,
                    "warning": "This column may be derived from the outcome (potential leakage)" if is_leakage else None
                })

            print(f"[Target Inference] LLM returned {len(candidates)} candidates, validated {len(validated_candidates)}")

            # ENSURE AT LEAST 3 CANDIDATES - supplement with other suitable columns if needed
            task_type = llm_response.get("task_type", "unknown")
            existing_cols = {c["column"] for c in validated_candidates}

            # ALWAYS ensure at least 3 candidates (unless EDA mode)
            if task_type != "eda":
                # Collect ALL valid columns first
                all_valid_cols = []
                for col_profile in profile.get("column_profiles", []):
                    col_name = col_profile["name"]
                    if col_name in existing_cols:
                        continue

                    unique_count = col_profile.get("unique_count", 0)
                    col_type = col_profile.get("column_type", "unknown")

                    # Skip ID-like columns
                    if is_id_like_column(col_name, unique_count, len(df)):
                        continue

                    # Skip constant columns
                    if col_profile.get("is_constant", False):
                        continue

                    # Build reason based on column type
                    if col_type == "numeric":
                        min_val = col_profile.get('min')
                        max_val = col_profile.get('max')
                        if min_val is not None and max_val is not None:
                            reason = f"Numeric column (range: {min_val:.0f} to {max_val:.0f})"
                        else:
                            reason = f"Numeric column ({unique_count} unique values)"
                        confidence = 0.4
                    else:
                        reason = f"Categorical column ({unique_count} categories)"
                        confidence = 0.3 if unique_count <= 20 else 0.2

                    all_valid_cols.append({
                        "column": col_name,
                        "confidence": confidence,
                        "reason": reason,
                        "col_type": col_type,
                        "is_leakage_risk": is_leakage_column(col_name),
                        "warning": None
                    })

                # Sort: numeric first (better for regression), then by confidence
                all_valid_cols.sort(key=lambda x: (0 if x["col_type"] == "numeric" else 1, -x["confidence"]))

                # Add candidates until we have at least 3
                for col in all_valid_cols:
                    if len(validated_candidates) >= 3:
                        break
                    validated_candidates.append(col)

            # Debug: Log how many candidates we have
            print(f"[Target Inference] Final candidate count: {len(validated_candidates)}")

            # Determine if user confirmation is needed
            chosen_target = None
            needs_confirmation = True

            if validated_candidates:
                best = validated_candidates[0]
                # Always require confirmation when there are multiple options
                if best.get("confidence", 0) >= 0.95 and len(validated_candidates) == 1:
                    # Very high confidence, single candidate - can auto-select
                    chosen_target = best["column"]
                    needs_confirmation = False
                else:
                    # Multiple candidates or moderate confidence - require selection
                    needs_confirmation = True

            result = {
                "task_type": task_type,
                "target_candidates": validated_candidates,
                "chosen_target": chosen_target,
                "needs_user_confirmation": needs_confirmation,
                "assumption_note": llm_response.get("assumption_note", "Please select a target column for prediction")
            }

            # Store in app state
            if chosen_target:
                app_state["target_column"] = chosen_target
            app_state["target_candidates"] = validated_candidates

            return make_response(True, result)

        else:
            # LLM unavailable - provide all non-ID columns as candidates for user selection
            fallback_candidates = []
            for col_profile in profile.get("column_profiles", []):
                col_name = col_profile["name"]
                unique_count = col_profile.get("unique_count", 0)

                # Skip ID-like columns
                if is_id_like_column(col_name, unique_count, len(df)):
                    continue

                # Skip constant columns
                if col_profile.get("is_constant", False):
                    continue

                col_type = col_profile.get("column_type", "unknown")
                fallback_candidates.append({
                    "column": col_name,
                    "confidence": 0.5,
                    "reason": f"{col_type} column with {unique_count} unique values",
                    "is_leakage_risk": is_leakage_column(col_name),
                    "warning": None
                })

            return make_response(True, {
                "task_type": "classification",  # Default to classification
                "target_candidates": fallback_candidates[:10],  # Limit to top 10
                "chosen_target": None,
                "needs_user_confirmation": True,
                "assumption_note": "Please select the column you want to predict"
            })

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


@app.post("/set-target")
async def set_target_column(request: SetTargetRequest):
    """Explicitly set the target column for prediction."""
    try:
        df = app_state.get("df_cleaned")
        if df is None:
            df = app_state.get("df")
        if df is None:
            return make_response(False, error="No dataset loaded")

        if request.target_column:
            if request.target_column not in df.columns:
                return make_response(False, error=f"Column '{request.target_column}' not found in dataset")
            app_state["target_column"] = request.target_column
        else:
            app_state["target_column"] = None

        return make_response(True, {
            "target_column": app_state.get("target_column"),
            "message": f"Target column set to: {request.target_column}" if request.target_column else "Target column cleared"
        })

    except Exception as e:
        return make_response(False, error=str(e))


@app.post("/plan")
async def generate_plan(request: PlanRequest):
    """Generate cleaning and visualization plan using LLM."""
    try:
        df = app_state.get("df")
        profile = app_state.get("profile")

        if df is None:
            return make_response(False, error="No dataset loaded")
        if profile is None:
            return make_response(False, error="Profile not generated. Call /profile first.")

        # Prepare context for LLM
        context = {
            "intent": request.intent,
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
            "numeric_cols": profile["numeric_cols"],
            "categorical_cols": profile["categorical_cols"],
            "datetime_cols": profile["datetime_cols"],
            "summary": profile["summary_text"],
            "column_profiles": [
                {k: v for k, v in p.items() if k != "top_values"}
                for p in profile["column_profiles"][:20]
            ],
        }

        # Optionally include sample rows
        if request.allow_sample_rows:
            sample = df.head(5).fillna("").to_dict(orient="records")
            context["sample_rows"] = sample

        # Build explicit column list for LLM
        all_columns = profile.get("numeric_cols", []) + profile.get("categorical_cols", []) + profile.get("datetime_cols", [])
        column_list = "\n".join([f"  - {col}" for col in all_columns])

        # Generate viz plan via LLM
        system_prompt = f"""You are a data science assistant. Given a dataset profile and user intent,
suggest exactly 7 visualizations that would be most insightful.

CRITICAL - USE THESE EXACT COLUMN NAMES (copy-paste them exactly):
{column_list}

RULES:
1. Use DIFFERENT chart types - max 2 of same type
2. Copy column names EXACTLY as listed above (case-sensitive!)
3. Match visualizations to user's question

Available chart types:
- missing_values_bar: {{}} (data quality - use once max)
- numeric_distribution: {{"column": "exact_col_name"}} (histogram)
- categorical_frequency: {{"column": "exact_col_name", "top_n": 10}} (bar chart)
- correlation_heatmap: {{}} (all numeric correlations - use once)
- scatter: {{"x_column": "col1", "y_column": "col2"}} (max 2 scatter charts)
- boxplot_outliers: {{"columns": ["col1", "col2"]}} (compare distributions)
- time_trend: {{"date_column": "date_col", "value_column": "num_col"}} (if datetime exists)
- grouped_bar: {{"group_column": "cat_col", "value_column": "num_col", "aggregation": "mean"}}

Good variety example: 1 missing_values, 1 correlation, 1-2 scatter, 1 boxplot, 1-2 distributions, 1 grouped_bar

Return JSON: {{"viz_plan": [...], "reasoning": "..."}}\""""

        user_prompt = f"User intent: {request.intent}\n\nDataset context:\n{json.dumps(context, indent=2, default=str)}"

        llm_response = await call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        if llm_response and "viz_plan" in llm_response:
            viz_plan = llm_response["viz_plan"][:7]
            reasoning = llm_response.get("reasoning", "")
        else:
            # Fallback to default plan
            viz_plan = generate_default_viz_plan(profile)
            reasoning = "Using default visualization plan (LLM unavailable)"

        # Ensure exactly 7 charts
        if len(viz_plan) < 7:
            default = generate_default_viz_plan(profile)
            viz_plan.extend(default[len(viz_plan):7])
        viz_plan = viz_plan[:7]

        app_state["viz_plan"] = viz_plan

        return make_response(True, {
            "viz_plan": viz_plan,
            "reasoning": reasoning,
        })

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


@app.post("/clean/apply")
async def apply_cleaning(request: CleanRequest):
    """Apply cleaning operations."""
    try:
        df = app_state.get("df")
        if df is None:
            return make_response(False, error="No dataset loaded")

        # Apply safe cleaning first (if not already done)
        if app_state.get("df_cleaned") is None:
            df_cleaned, safe_log = apply_safe_cleaning(df)
            app_state["df_cleaned"] = df_cleaned
            app_state["cleaning_log"] = safe_log
        else:
            df_cleaned = app_state["df_cleaned"]

        # Apply recommended cleaning if requested
        if request.apply_recommended:
            profile = app_state.get("profile")
            if profile is None:
                # Re-profile cleaned data
                profile = profile_dataframe(df_cleaned)

            df_cleaned, rec_log = apply_recommended_cleaning(df_cleaned, profile)
            app_state["df_cleaned"] = df_cleaned
            app_state["cleaning_log"].extend(rec_log)

        # Save cleaned dataset
        cleaned_path = CLEANED_DIR / f"cleaned_{app_state['dataset_name'] or 'dataset'}.csv"
        df_cleaned.to_csv(cleaned_path, index=False)

        # Update profile
        app_state["profile"] = profile_dataframe(df_cleaned)

        # Preview
        preview = df_cleaned.head(10).fillna("").to_dict(orient="records")

        return make_response(True, {
            "cleaning_log": app_state["cleaning_log"],
            "n_rows": len(df_cleaned),
            "n_cols": len(df_cleaned.columns),
            "columns": list(df_cleaned.columns),  # Include new column names for frontend
            "preview": preview,
            "profile_summary": app_state["profile"]["summary_text"],
            "column_name_mapping": app_state.get("column_name_mapping", {}),  # Original -> snake_case mapping
        })

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


@app.post("/charts")
async def generate_charts(request: ChartsRequest):
    """Generate exactly 7 unique visualizations with deduplication."""
    try:
        df = app_state.get("df_cleaned")
        if df is None:
            df = app_state.get("df")
        if df is None:
            return make_response(False, error="No dataset loaded")

        profile = app_state.get("profile")
        if profile is None:
            profile = profile_dataframe(df)
            app_state["profile"] = profile

        viz_plan = app_state.get("viz_plan")
        if not viz_plan:
            viz_plan = generate_default_viz_plan(profile)
            app_state["viz_plan"] = viz_plan

        # Clear old charts
        for f in CHARTS_DIR.glob("*.png"):
            f.unlink()

        # Get column name mapping for translation (original -> cleaned names)
        col_mapping = app_state.get("column_name_mapping", {})
        df_columns = list(df.columns)  # Pass actual DataFrame columns for fuzzy matching

        # Chart deduplication: track signatures to prevent duplicates
        seen_signatures = set()
        chart_results = []
        chart_index = 1

        # First pass: process viz_plan with deduplication
        for chart_spec in viz_plan:
            if len(chart_results) >= 7:
                break

            chart_type = chart_spec.get("type", "missing_values_bar")
            config = chart_spec.get("config", {})

            # Translate column names from original to cleaned names (with fuzzy matching)
            translated_config = translate_chart_config(config, col_mapping, df_columns)

            # Generate signature for deduplication
            signature = get_chart_signature(chart_type, translated_config)

            # Skip if duplicate
            if signature in seen_signatures:
                continue

            seen_signatures.add(signature)
            output_path = CHARTS_DIR / f"chart_{chart_index}_{chart_type}.png"

            stats = generate_chart(chart_type, df, profile, translated_config, output_path)
            stats["index"] = chart_index
            stats["filename"] = output_path.name
            stats["signature"] = signature
            chart_results.append(stats)
            chart_index += 1

        # Second pass: backfill with default charts if we have less than 7
        if len(chart_results) < 7:
            default_plan = generate_default_viz_plan(profile)
            for chart_spec in default_plan:
                if len(chart_results) >= 7:
                    break

                chart_type = chart_spec.get("type", "missing_values_bar")
                config = chart_spec.get("config", {})
                translated_config = translate_chart_config(config, col_mapping, df_columns)
                signature = get_chart_signature(chart_type, translated_config)

                if signature in seen_signatures:
                    continue

                seen_signatures.add(signature)
                output_path = CHARTS_DIR / f"chart_{chart_index}_{chart_type}.png"

                stats = generate_chart(chart_type, df, profile, translated_config, output_path)
                stats["index"] = chart_index
                stats["filename"] = output_path.name
                stats["signature"] = signature
                chart_results.append(stats)
                chart_index += 1

        # Third pass: if still under 7, add additional numeric distributions
        numeric_cols = profile.get("numeric_cols", [])
        col_idx = 0
        while len(chart_results) < 7 and col_idx < len(numeric_cols):
            col = numeric_cols[col_idx]
            config = {"column": col}
            signature = get_chart_signature("numeric_distribution", config)

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                output_path = CHARTS_DIR / f"chart_{chart_index}_numeric_distribution.png"
                stats = generate_chart("numeric_distribution", df, profile, config, output_path)
                stats["index"] = chart_index
                stats["filename"] = output_path.name
                stats["signature"] = signature
                chart_results.append(stats)
                chart_index += 1

            col_idx += 1

        app_state["chart_stats"] = {r["index"]: r for r in chart_results}

        return make_response(True, {
            "charts": chart_results,
            "chart_dir": str(CHARTS_DIR),
        })

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


def get_chart_signature(chart_type: str, config: Dict) -> str:
    """Generate unique signature for chart deduplication."""
    cols = []
    for key in ['column', 'x_column', 'y_column', 'date_column', 'value_column', 'group_column']:
        if key in config and config[key]:
            cols.append(config[key])
    if 'columns' in config and isinstance(config['columns'], list):
        cols.extend(config['columns'])
    cols_str = "_".join(sorted(set(cols)))
    agg = config.get('aggregation', '')
    return f"{chart_type}|{cols_str}|{agg}"


def determine_relevance(chart_type: str, config: Dict, intent: str) -> tuple:
    """Determine relevance of chart to user intent."""
    intent_lower = intent.lower()

    # Keywords for different relevance patterns
    outcome_keywords = ['outcome', 'result', 'win', 'predict', 'factor', 'affect', 'impact', 'determine']
    temporal_keywords = ['time', 'trend', 'over time', 'temporal', 'date', 'when']
    quality_keywords = ['quality', 'missing', 'clean', 'data quality']

    is_outcome_focused = any(kw in intent_lower for kw in outcome_keywords)
    is_temporal_focused = any(kw in intent_lower for kw in temporal_keywords)
    is_quality_focused = any(kw in intent_lower for kw in quality_keywords)

    # Determine relevance based on chart type and intent
    if chart_type == "missing_values_bar":
        if is_quality_focused:
            return ("High", "Data quality is central to your analysis goal")
        return ("Low", "Data quality overview, not directly addressing the main question")

    elif chart_type == "correlation_heatmap":
        if is_outcome_focused:
            return ("High", "Shows relationships between features relevant to outcomes")
        return ("Medium", "Useful for understanding feature relationships")

    elif chart_type == "scatter":
        if is_outcome_focused:
            return ("High", "Visualizes potential relationships between key variables")
        return ("Medium", "Exploratory view of variable relationships")

    elif chart_type == "grouped_bar":
        if is_outcome_focused:
            return ("High", "Compares metrics across categories relevant to outcomes")
        return ("Medium", "Shows categorical comparisons")

    elif chart_type == "time_trend":
        if is_temporal_focused:
            return ("High", "Directly addresses temporal analysis goal")
        return ("Low", "Temporal view, may not directly address the question")

    elif chart_type in ["numeric_distribution", "boxplot_outliers"]:
        return ("Medium", "Provides context on data distribution and outliers")

    elif chart_type == "categorical_frequency":
        return ("Medium", "Shows category distribution in the data")

    return ("Medium", "Exploratory visualization")


@app.post("/explain")
async def explain_charts(request: ExplainRequest):
    """Generate explanations for charts."""
    try:
        chart_stats = app_state.get("chart_stats", {})
        if not chart_stats:
            return make_response(False, error="No charts generated. Call /charts first.")

        # Check if target column exists (for language safety)
        target_col = app_state.get("target_column")
        has_model = app_state.get("has_trained_model", False)

        explanations = []

        for idx, stats in chart_stats.items():
            chart_type = stats.get("type", "unknown")
            config = stats.get("config", {})

            # Determine relevance
            relevance, relevance_reason = determine_relevance(chart_type, config, request.intent)

            # Extract numeric facts for grounding requirement
            numeric_facts = []
            for key in ['correlation', 'mean', 'median', 'std', 'skewness', 'outlier_pct',
                        'max_correlation', 'n_points', 'total_missing', 'cols_with_missing',
                        'top_count', 'unique_shown', 'top_value']:
                if key in stats and stats[key] is not None:
                    numeric_facts.append(f"{key}: {stats[key]}")

            facts_str = ", ".join(numeric_facts[:5]) if numeric_facts else "No numeric stats available"

            # Build safe language rules based on context
            safety_rules = """
CRITICAL LANGUAGE RULES (MANDATORY - VIOLATION IS FAILURE):

FORBIDDEN PHRASES (NEVER USE):
- "causes", "drives", "leads to", "results in"
- "determines outcome", "determines the result"
- "most important factor", "key factor", "primary driver"
- "predicts", "predictive of", "indicator of success/failure"
- "winning/losing teams have", "successful entities tend to"
- Any implication that a predictive model exists when it doesn't
- Any domain-specific generalizations (e.g., "teams", "matches", "sales")

REQUIRED PHRASES (USE THESE INSTEAD):
- "associated with", "correlated with" (with correlation value)
- "may indicate", "appears to show"
- "shows a pattern", "exhibits variation"
- "tends to co-occur with"
- "exploratory observation"

MANDATORY STATEMENTS:
- "This is exploratory data analysis (EDA), not causal inference"
- Do NOT extrapolate beyond this specific dataset
"""
            if not target_col and not has_model:
                safety_rules += """
STRICT MODE (NO TARGET/NO MODEL):
No confirmed target column and no trained predictive model exist.
- Do NOT use causal, predictive, or importance language
- Do NOT imply any variable 'affects', 'impacts', or 'influences' outcomes
- Describe ONLY what the data shows, not what it might mean for outcomes
- This is PURELY exploratory pattern description
"""

            # Special rules for correlation heatmaps
            heatmap_rules = ""
            if chart_type == "correlation_heatmap":
                heatmap_rules = """
CORRELATION HEATMAP SPECIFIC RULES (MANDATORY):
- Describe ONLY feature-to-feature relationships (multicollinearity, redundancy, coupling)
- NEVER mention: "importance", "impact on outcome", "key factor for outcome", "predicts"
- Correlation heatmaps show relationships BETWEEN FEATURES, NOT feature-to-outcome relationships
- Focus on: which features move together, potential multicollinearity issues, redundant features
- Good example: "Features X and Y show strong positive correlation (r=0.72), suggesting redundancy"
- Bad example: "X is the most important factor for the outcome"
"""

            # Grounding requirement
            grounding_rule = f"""
NUMERIC GROUNDING REQUIREMENT (MANDATORY):
- You MUST cite at least TWO specific numeric values from: {facts_str}
- If insufficient numeric signal exists, say: "Insufficient statistical signal to draw conclusions"
- Do NOT describe patterns or trends without numeric support

NO HALLUCINATION RULE (STRICT):
- ONLY reference statistics explicitly provided in the stats
- Do NOT invent, approximate, or imply statistics not given
- Do NOT reference aggregates not computed (e.g., "average of winning rows")
- If a statistic doesn't exist, do NOT mention it
"""

            system_prompt = f"""You are a friendly data analyst explaining charts in plain English. Write naturally, like you're helping a colleague understand what they're looking at.

{safety_rules}
{heatmap_rules}
{grounding_rule}

STYLE:
- Write conversationally, not robotically
- When citing numbers, briefly explain what they mean in plain terms
- Keep it concise but informative
- No markdown formatting (no ** or ## symbols)"""

            if request.mode == "quick":
                prompt = f"""Explain this {chart_type} chart in 2-3 clear sentences that anyone could understand.

Stats: {json.dumps(stats, default=str)}
User's analysis goal: {request.intent}

Start with "[{relevance} relevance] " then explain what the chart shows.
Include at least 1-2 specific numbers and what they mean in simple terms."""
            else:
                prompt = f"""Explain this {chart_type} chart in 4-5 sentences that anyone could understand.

Stats: {json.dumps(stats, default=str)}
User's analysis goal: {request.intent}

Start with "[{relevance} relevance] {relevance_reason}"

Then explain:
- What the chart is showing (in plain terms)
- The key patterns with specific numbers (explain what the numbers mean)
- Any caveats to keep in mind
Keep it conversational and easy to read."""

            llm_response = await call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ], json_mode=False)

            if llm_response and "text" in llm_response:
                explanation = llm_response["text"]
            else:
                # Fallback explanation with natural language
                explanation = f"[{relevance} relevance] {relevance_reason}\n\n"
                explanation += f"This {chart_type.replace('_', ' ')} shows "
                if "correlation" in stats:
                    corr = stats['correlation']
                    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                    direction = "positive" if corr > 0 else "negative"
                    explanation += f"a {strength} {direction} relationship (correlation: {corr:.2f}). "
                if "mean" in stats and "median" in stats:
                    explanation += f"The average is {stats['mean']:.1f} with a median of {stats['median']:.1f}. "
                elif "mean" in stats:
                    explanation += f"the average value is {stats['mean']:.1f}. "
                if "top_category" in stats and "top_count" in stats:
                    explanation += f"The most common category is '{stats['top_category']}' appearing {stats['top_count']} times. "
                elif "top_category" in stats:
                    explanation += f"The most common category is '{stats['top_category']}'. "
                explanation += "This is exploratory - it shows patterns but doesn't prove cause and effect."

            explanations.append({
                "chart_index": idx,
                "chart_type": chart_type,
                "explanation": explanation,
                "relevance": relevance,
                "relevance_reason": relevance_reason,
                "stats_summary": stats,
            })

        return make_response(True, {"explanations": explanations})

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


@app.post("/cleaning-summary")
async def get_cleaning_summary():
    """Return detailed summary of what was cleaned."""
    try:
        cleaning_log = app_state.get("cleaning_log", [])
        col_mapping = app_state.get("column_name_mapping", {})

        # Build summary of column renames
        columns_renamed = [
            {"original": orig, "cleaned": clean}
            for orig, clean in col_mapping.items() if orig != clean
        ]

        summary = {
            "columns_renamed": columns_renamed,
            "columns_renamed_count": len(columns_renamed),
            "actions": cleaning_log,
            "total_actions": len(cleaning_log),
        }

        # Add before/after stats
        df_original = app_state.get("df")
        df_cleaned = app_state.get("df_cleaned")

        if df_original is not None:
            summary["before"] = {
                "rows": len(df_original),
                "columns": len(df_original.columns),
                "missing_total": int(df_original.isna().sum().sum()),
                "duplicates": int(df_original.duplicated().sum()),
                "column_names": list(df_original.columns),
            }

        if df_cleaned is not None:
            summary["after"] = {
                "rows": len(df_cleaned),
                "columns": len(df_cleaned.columns),
                "missing_total": int(df_cleaned.isna().sum().sum()),
                "duplicates": int(df_cleaned.duplicated().sum()),
                "column_names": list(df_cleaned.columns),
            }

        # Calculate changes
        if df_original is not None and df_cleaned is not None:
            summary["changes"] = {
                "rows_removed": len(df_original) - len(df_cleaned),
                "missing_values_changed": int(df_original.isna().sum().sum()) - int(df_cleaned.isna().sum().sum()),
                "duplicates_removed": int(df_original.duplicated().sum()) - int(df_cleaned.duplicated().sum()),
            }

        return make_response(True, summary)

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


class FinalSummaryRequest(BaseModel):
    intent: str
    allow_sample_rows: bool = False


@app.post("/final-summary")
async def generate_final_summary(request: FinalSummaryRequest):
    """Generate comprehensive LLM summary answering the user's original question."""
    try:
        profile = app_state.get("profile")
        chart_stats = app_state.get("chart_stats", {})
        cleaning_log = app_state.get("cleaning_log", [])

        df = app_state.get("df_cleaned")
        if df is None:
            df = app_state.get("df")

        if df is None:
            return make_response(False, error="No dataset loaded")

        # Determine if this is model-based or EDA-based
        has_model = app_state.get("has_trained_model", False)
        model_results = app_state.get("model_results")
        target_col = app_state.get("target_column")

        # Build comprehensive context for the LLM (NO RAW DATA)
        context = {
            "user_question": request.intent,
            "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
            "profile_summary": profile.get("summary_text", "") if profile else "",
            "numeric_columns": profile.get("numeric_cols", []) if profile else [],
            "categorical_columns": profile.get("categorical_cols", []) if profile else [],
            "charts_generated": [s.get("type") for s in chart_stats.values()],
            "key_stats_from_charts": [
                {
                    "chart": s.get("type"),
                    "stats": {k: v for k, v in s.items() if k not in ["path", "config", "success", "index", "filename", "signature"]}
                }
                for s in chart_stats.values() if s.get("success")
            ],
            "cleaning_actions_count": len(cleaning_log),
            "analysis_type": "MODEL-BASED" if has_model else "EDA-ONLY (Exploratory)",
            "has_trained_model": has_model,
            "target_column": target_col,
        }

        # Add model results if available
        if has_model and model_results:
            context["model_info"] = {
                "model_type": model_results["model_summary"]["model_type"],
                "task_type": model_results["task_type"],
                "metrics": model_results["metrics"],
                "train_rows": model_results["model_summary"]["train_rows"],
                "test_rows": model_results["model_summary"]["test_rows"],
                "top_features": model_results["model_summary"]["feature_importance"][:10],
            }

            # Check for potential leakage features
            leakage_suspects = []
            target_lower = target_col.lower() if target_col else ""
            for feat in model_results["model_summary"]["feature_importance"][:5]:
                feat_name = feat["feature"].lower()
                # Check if feature name contains outcome-related terms
                if any(term in feat_name for term in ["result", "outcome", "score", "win", "loss", "target", target_lower]):
                    leakage_suspects.append(feat["feature"])
            if leakage_suspects:
                context["leakage_warning"] = f"Features that may encode target information: {leakage_suspects}"

        # Build the system prompt for natural, readable output
        system_prompt = """You are a friendly data science expert explaining analysis results in plain, conversational English.

WRITING STYLE:
- Write naturally, like explaining to a smart colleague who isn't a data scientist
- NO markdown formatting (no ** or ## symbols - they render as literal text)
- Use simple dashes (-) for lists, with proper spacing
- Use clear paragraph breaks between sections
- Include brief plain-English explanations of technical terms in parentheses
- Be warm and clear, not robotic or overly formal

STRUCTURE (flow naturally between these parts):

Start by restating their question in your own words to show you understood it.

Then explain what type of analysis you did:
- If a model was trained: briefly explain what the model does in simple terms
- If EDA only: explain you explored the data to find patterns (no predictions made)

Share your findings with actual numbers, and for each technical metric add a simple interpretation:
- Example: "The model got 70% accuracy, meaning it correctly predicted the outcome about 7 times out of 10."
- Example: "The F1 score of 0.41 suggests the model struggles somewhat with certain categories."
- If model: mention which features (variables) the model relied on most for its predictions

Include important caveats in plain language:
- How big was the dataset and why that matters
- If model: this is a starting point, not a finished product
- If EDA: finding patterns doesn't prove cause-and-effect
- If there's a leakage warning: explain it simply
- These results apply to this specific data only

End with a clear bottom line - 2-3 sentences a non-technical person could understand.

LANGUAGE RULES:
- Avoid: "causes", "drives", "determines", "key factor" (unless describing model features specifically)
- Use instead: "associated with", "the model relied on", "patterns suggest", "we observed"
- When mentioning important features, always clarify "in this model" or "on this dataset"
- No broad generalizations beyond this specific dataset
"""

        user_prompt = f"""Write a clear, readable summary of this analysis for someone who isn't a data scientist.

ANALYSIS CONTEXT:
{json.dumps(context, indent=2, default=str)}

REQUIREMENTS:
- Write in plain, conversational English (no markdown symbols like ** or ##)
- Include at least 3 specific numbers from the data
- For each technical term or metric, add a simple explanation
- If has_trained_model is true: explain what features the model found most useful
- If has_trained_model is false: focus on patterns observed, noting this is exploratory
- Include the caveats (dataset size, limitations, etc.)
- If leakage_warning exists: explain it in simple terms
- End with a clear takeaway
- Keep it 250-400 words, easy to scan"""

        llm_response = await call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], json_mode=False)

        if llm_response and "text" in llm_response:
            summary_text = llm_response["text"]
        else:
            # Fallback summary with natural, readable format (no markdown)
            summary_text = f"""You asked: "{request.intent}"

"""
            if has_model and model_results:
                model_type = model_results['model_summary']['model_type']
                summary_text += f"""What we did: We trained a {model_type} model to analyze your data. This type of model looks for patterns in the data to make predictions.

"""
                if model_results["task_type"] == "classification":
                    acc = model_results['metrics']['accuracy']
                    f1 = model_results['metrics']['macro_f1']
                    acc_simple = int(acc * 10)
                    summary_text += f"""What we found:
- The model achieved {acc:.1%} accuracy (meaning it correctly predicted the outcome about {acc_simple} times out of 10)
- The F1 score was {f1:.1%} (this measures how well the model balances finding all relevant cases vs. being precise)
- We analyzed {context['dataset_shape']['rows']} records with {context['dataset_shape']['columns']} different variables
"""
                else:
                    r2 = model_results['metrics']['r2']
                    rmse = model_results['metrics']['rmse']
                    r2_pct = int(r2 * 100)
                    summary_text += f"""What we found:
- The model achieved an R² of {r2:.3f} (meaning it explains about {r2_pct}% of the variation in the data)
- The prediction error (RMSE) was {rmse:.2f}
- We analyzed {context['dataset_shape']['rows']} records with {context['dataset_shape']['columns']} different variables
"""
                # Top features
                top_features = model_results["model_summary"]["feature_importance"][:3]
                if top_features:
                    feature_names = ", ".join([f["feature"] for f in top_features])
                    summary_text += f"""
Features the model relied on most: In this model, the variables that had the biggest influence on predictions were: {feature_names}. This tells us what the model "paid attention to" - though it doesn't necessarily mean these cause the outcome.
"""
            else:
                summary_text += f"""What we did: We performed exploratory data analysis (EDA) - this means we looked for patterns and relationships in your data without building a predictive model.

What we found:
- Analyzed {context['dataset_shape']['rows']} records with {context['dataset_shape']['columns']} different variables
- Generated {len(chart_stats)} visualizations to explore the data
- Examined {len(context.get('numeric_columns', []))} numeric and {len(context.get('categorical_columns', []))} categorical columns
"""

            # Caveats
            summary_text += f"""
Important caveats:
- This analysis is based on {context['dataset_shape']['rows']} records - {"a relatively small sample, so results should be validated with more data" if context['dataset_shape']['rows'] < 100 else "keep in mind results are specific to this dataset"}
"""
            if has_model:
                summary_text += "- This is a baseline model, meaning it's a starting point for understanding patterns, not a production-ready system\n"
            else:
                summary_text += "- Finding patterns doesn't prove cause-and-effect - correlation is not causation\n"

            if context.get("leakage_warning"):
                summary_text += f"- Heads up: {context['leakage_warning']} - this could make results look better than they'd be in practice\n"

            summary_text += "- These findings apply to this specific dataset and may not generalize to other data\n"

            # Bottom line
            summary_text += f"""
Bottom line: {"This model gives us a baseline understanding of what patterns exist in your data. The features it relied on most give clues about what variables are associated with the outcome, but further validation would be needed before using this for real decisions." if has_model else "This exploratory analysis reveals some interesting patterns in your data. To draw stronger conclusions or make predictions, consider building a predictive model with a confirmed target variable."}
"""

        return make_response(True, {
            "summary": summary_text,
            "context_used": context,
            "analysis_type": "model-based" if has_model else "eda-only"
        })

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


@app.get("/download/cleaned")
async def download_cleaned():
    """Download cleaned dataset as CSV."""
    try:
        cleaned_files = list(CLEANED_DIR.glob("*.csv"))
        if not cleaned_files:
            return JSONResponse(
                status_code=404,
                content=make_response(False, error="No cleaned dataset available")
            )

        # Return most recent
        latest = max(cleaned_files, key=lambda p: p.stat().st_mtime)
        return FileResponse(
            path=str(latest),
            filename=latest.name,
            media_type="text/csv"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=make_response(False, error=str(e))
        )


@app.get("/download/charts")
async def download_charts():
    """Download all charts as ZIP."""
    try:
        chart_files = list(CHARTS_DIR.glob("*.png"))
        if not chart_files:
            return JSONResponse(
                status_code=404,
                content=make_response(False, error="No charts available")
            )

        # Create ZIP
        zip_path = OUTPUTS_DIR / "charts.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for chart_file in chart_files:
                zf.write(chart_file, chart_file.name)

        return FileResponse(
            path=str(zip_path),
            filename="datanarrate_charts.zip",
            media_type="application/zip"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=make_response(False, error=str(e))
        )


@app.get("/chart/{filename}")
async def get_chart(filename: str):
    """Get a specific chart image."""
    chart_path = CHARTS_DIR / filename
    if not chart_path.exists():
        return JSONResponse(
            status_code=404,
            content=make_response(False, error="Chart not found")
        )

    return FileResponse(
        path=str(chart_path),
        media_type="image/png"
    )


# ---------------------
# Modeling Endpoints
# ---------------------

@app.post("/model/train")
async def train_model(request: ModelTrainRequest):
    """Train a classical ML model on the dataset."""
    try:
        df = app_state.get("df_cleaned")
        if df is None:
            df = app_state.get("df")
        if df is None:
            return make_response(False, error="No dataset loaded")

        target_col = request.target_column
        if target_col not in df.columns:
            return make_response(False, error=f"Target column '{target_col}' not found in dataset")

        # Store target in app_state
        app_state["target_column"] = target_col

        # Get profile for column types
        profile = app_state.get("profile")
        if profile is None:
            profile = profile_dataframe(df)
            app_state["profile"] = profile

        # Prepare features and target - drop rows with missing target
        df_model = df.dropna(subset=[target_col]).copy()
        if len(df_model) < 10:
            return make_response(False, error=f"Insufficient data after dropping missing targets ({len(df_model)} rows)")

        y = df_model[target_col]
        X = df_model.drop(columns=[target_col])

        # Get numeric and categorical columns (excluding target)
        numeric_cols = [c for c in profile.get("numeric_cols", []) if c != target_col and c in X.columns]
        categorical_cols = [c for c in profile.get("categorical_cols", []) if c != target_col and c in X.columns]

        # Filter to only include columns that exist in X
        numeric_cols = [c for c in numeric_cols if c in X.columns]
        categorical_cols = [c for c in categorical_cols if c in X.columns]

        if not numeric_cols and not categorical_cols:
            return make_response(False, error="No valid feature columns found for modeling")

        # Build preprocessor
        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)

        # Import sklearn components
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_seed
        )

        # Determine task type - ALWAYS validate against actual data, don't just trust LLM
        task_type = request.task_type
        n_unique = y.nunique()
        n_samples = len(y)
        is_numeric_target = pd.api.types.is_numeric_dtype(y)

        # Smart task type detection based on actual data characteristics
        if is_numeric_target:
            # For numeric targets, check if it looks like classification or regression
            unique_ratio = n_unique / n_samples if n_samples > 0 else 0

            if n_unique <= 10:
                # Few unique values = classification (even if numeric, like 0/1 or 1/2/3)
                task_type = "classification"
            elif unique_ratio > 0.5:
                # More than 50% unique values = definitely regression
                task_type = "regression"
            elif n_unique > 20:
                # Many unique numeric values = regression
                task_type = "regression"
            else:
                # 10-20 unique values, use LLM's suggestion or default to classification
                if task_type not in ["classification", "regression"]:
                    task_type = "classification"
        else:
            # Categorical target = classification
            task_type = "classification"

        print(f"[Model] Target '{request.target_column}': {n_unique} unique values, {n_samples} samples, numeric={is_numeric_target} -> task_type={task_type}")

        # Build and train model
        if task_type == "classification":
            # Check for class imbalance
            class_counts = y_train.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else 1
            class_weight = "balanced" if imbalance_ratio > 2 else None

            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    max_iter=request.max_iter,
                    class_weight=class_weight,
                    random_state=request.random_seed,
                    solver='lbfgs'
                ))
            ])
        else:
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', Ridge(random_state=request.random_seed))
            ])

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics and generate artifacts
        if task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

            # Get unique labels
            labels = list(model.named_steps['classifier'].classes_)

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "labels": [str(l) for l in labels]
            }

            # Generate confusion matrix plot
            artifact_path = MODELING_DIR / "confusion_matrix.png"
            generate_confusion_matrix_plot(y_test, y_pred, [str(l) for l in labels], artifact_path)
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }

            # Generate residuals plot
            artifact_path = MODELING_DIR / "residuals.png"
            generate_residuals_plot(y_test, y_pred, artifact_path)

        # Extract feature importance
        feature_importance = []
        try:
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()

            if task_type == "classification":
                coefs = model.named_steps['classifier'].coef_
                # For multi-class, take mean absolute value across classes
                if len(coefs.shape) == 2:
                    mean_coefs = np.abs(coefs).mean(axis=0)
                else:
                    mean_coefs = np.abs(coefs)

                for name, coef in sorted(zip(feature_names, mean_coefs), key=lambda x: -abs(x[1]))[:20]:
                    feature_importance.append({"feature": str(name), "coefficient": float(coef)})
            else:
                coefs = model.named_steps['regressor'].coef_
                for name, coef in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1]))[:20]:
                    feature_importance.append({"feature": str(name), "coefficient": float(coef)})

        except Exception as e:
            print(f"Failed to extract feature importance: {e}")

        # Store in app_state
        app_state["model"] = model
        app_state["has_trained_model"] = True
        app_state["model_results"] = {
            "task_type": task_type,
            "target_column": target_col,
            "metrics": metrics,
            "artifacts": {"filename": artifact_path.name, "path": str(artifact_path)},
            "model_summary": {
                "model_type": "LogisticRegression" if task_type == "classification" else "Ridge",
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "n_features_encoded": len(feature_names) if 'feature_names' in locals() else 0,
                "feature_importance": feature_importance,
            }
        }

        return make_response(True, app_state["model_results"])

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


@app.post("/model/explain")
async def explain_model(request: ModelExplainRequest):
    """Generate LLM explanation of modeling results."""
    try:
        model_results = app_state.get("model_results")
        if model_results is None:
            return make_response(False, error="No model trained. Call /model/train first.")

        target_col = model_results["target_column"]

        # Check for potential leakage in top features
        leakage_suspects = []
        target_lower = target_col.lower() if target_col else ""
        for feat in model_results["model_summary"]["feature_importance"][:10]:
            feat_name = feat["feature"].lower()
            # Check if feature name contains outcome-related terms
            if any(term in feat_name for term in ["result", "outcome", "score", "win", "loss", "target", "label", target_lower]):
                leakage_suspects.append(feat["feature"])

        # Build context for LLM (no raw data!)
        context = {
            "task_type": model_results["task_type"],
            "target_column": target_col,
            "metrics": model_results["metrics"],
            "model_type": model_results["model_summary"]["model_type"],
            "train_rows": model_results["model_summary"]["train_rows"],
            "test_rows": model_results["model_summary"]["test_rows"],
            "n_features": model_results["model_summary"]["n_features_encoded"],
            "top_features": model_results["model_summary"]["feature_importance"][:10],
            "potential_leakage_features": leakage_suspects,
        }

        # Assess quality based on metrics
        if model_results["task_type"] == "classification":
            accuracy = model_results["metrics"]["accuracy"]
            quality = "Good" if accuracy >= 0.75 else "Medium" if accuracy >= 0.5 else "Weak"
        else:
            r2 = model_results["metrics"]["r2"]
            quality = "Good" if r2 >= 0.5 else "Medium" if r2 >= 0.2 else "Weak"

        system_prompt = """You are a data science expert explaining machine learning model results.

STRICT LANGUAGE DISCIPLINE (MANDATORY):

ALLOWED PHRASES:
- "The model relied more heavily on..."
- "Higher feature importance values indicate model reliance on..."
- "In this trained model..."
- "Based on this dataset and model..."
- "The model assigns higher weight to..."
- "Model coefficient magnitude suggests..."

FORBIDDEN PHRASES (NEVER USE):
- "Factors that contribute to outcome"
- "Entities with higher X tend to have Y"
- "Key factors for success/failure"
- "Causes", "drives", "leads to", "determines"
- Any domain-general claims
- Any use of correlation stats to justify model behavior

MODEL-SCOPE RULE:
Explanations may ONLY reference:
- Trained model outputs (coefficients, predictions)
- Feature importance values from THIS model
- Model metrics (accuracy, F1, R², RMSE)
- Dataset size (train/test rows)

NO HALLUCINATION:
- Only cite metrics explicitly provided
- Do NOT invent domain interpretations
- Do NOT extrapolate beyond this model/dataset

LEAKAGE DISCLOSURE (IF applicable):
If potential_leakage_features is non-empty, you MUST:
- Warn: "Features [X] may introduce target leakage"
- State: "Feature importance for these may be artificially inflated"
- Advise: "Interpret with caution; consider removing these features"

Return STRICT JSON:
{
  "summary": "2-4 sentences about model performance using ONLY allowed phrases",
  "quality_assessment": "Good|Medium|Weak",
  "key_drivers": [{"feature": "...", "effect": "positive|negative", "magnitude": 0.5}],
  "warnings": ["..."],
  "next_actions": ["..."]
}"""

        user_prompt = f"""Explain this {model_results['task_type']} model using STRICT modeling language.

Context (ONLY reference these values):
{json.dumps(context, indent=2, default=str)}

User's analysis goal: {request.intent}

REQUIREMENTS:
1. Reference at least TWO specific metrics
2. State this is a BASELINE model
3. Use ONLY model-scoped language
4. If leakage suspects exist, include mandatory warning
5. No domain generalizations

Generate JSON summary."""

        llm_response = await call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # Build warnings list
        base_warnings = [
            "This is a baseline model - results require validation with domain expertise",
            "Feature importance shows model reliance, not real-world causation"
        ]
        if leakage_suspects:
            base_warnings.insert(0, f"POTENTIAL LEAKAGE: Features {leakage_suspects} may encode target information. Importance values may be artificially inflated.")

        if llm_response:
            # Ensure required fields exist and add leakage warnings
            explanation = {
                "mode": request.mode,
                "summary": llm_response.get("summary", f"This {model_results['task_type']} baseline model achieved {quality.lower()} performance."),
                "quality_assessment": llm_response.get("quality_assessment", quality),
                "key_drivers": llm_response.get("key_drivers", []),
                "warnings": llm_response.get("warnings", []) + [w for w in base_warnings if w not in llm_response.get("warnings", [])],
                "next_actions": llm_response.get("next_actions", ["Validate with domain expertise", "Try cross-validation", "Consider feature engineering"]),
                "potential_leakage": leakage_suspects,
            }
        else:
            # Fallback explanation
            if model_results["task_type"] == "classification":
                metrics_str = f"accuracy of {model_results['metrics']['accuracy']:.1%} and F1 score of {model_results['metrics']['macro_f1']:.1%}"
            else:
                metrics_str = f"R² of {model_results['metrics']['r2']:.3f} and RMSE of {model_results['metrics']['rmse']:.2f}"

            explanation = {
                "mode": request.mode,
                "summary": f"This {model_results['task_type']} baseline model ({model_results['model_summary']['model_type']}) achieved {metrics_str}. The model was trained on {model_results['model_summary']['train_rows']} samples and tested on {model_results['model_summary']['test_rows']}. Feature importance values indicate model reliance, not causal relationships.",
                "quality_assessment": quality,
                "key_drivers": [
                    {"feature": f["feature"], "effect": "positive" if f["coefficient"] > 0 else "negative", "magnitude": abs(f["coefficient"])}
                    for f in model_results["model_summary"]["feature_importance"][:5]
                ],
                "warnings": base_warnings,
                "next_actions": [
                    "Validate results with domain expertise",
                    "Consider feature engineering",
                    "Try cross-validation for more robust estimates"
                ],
                "potential_leakage": leakage_suspects,
            }

        return make_response(True, explanation)

    except Exception as e:
        traceback.print_exc()
        return make_response(False, error=str(e))


@app.get("/model/artifact/{filename}")
async def get_model_artifact(filename: str):
    """Get a model artifact (confusion matrix, residuals plot)."""
    artifact_path = MODELING_DIR / filename
    if not artifact_path.exists():
        return JSONResponse(
            status_code=404,
            content=make_response(False, error="Artifact not found")
        )

    return FileResponse(
        path=str(artifact_path),
        media_type="image/png"
    )


# ---------------------
# Main Entry Point
# ---------------------

if __name__ == "__main__":
    import sys
    import uvicorn

    # Port from command-line argument (passed by ContextUI) or fallback to env/default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else SERVER_PORT

    print(f"Starting DataNarrate server at http://{SERVER_HOST}:{port}")
    print(f"Sample dataset: {SAMPLE_DATASET_CSV if SAMPLE_DATASET_CSV.exists() else SAMPLE_DATASET_XLSX}")
    uvicorn.run(app, host=SERVER_HOST, port=port)
