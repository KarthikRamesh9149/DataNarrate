"""
DataNarrate Streamlit App - Intent-Aware Data Science Copilot.

This app reuses the existing DataNarrate analytics engine and presents it as a
single polished Streamlit workflow: load data, profile quality, clean, visualize,
train a baseline model, and export results.
"""

import asyncio
import copy
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

import datanarrate_server as engine


APP_TITLE = "DataNarrate"
APP_SUBTITLE = "Intent-aware data science copilot for profiling, cleaning, visualization, and baseline ML."
SUPPORTED_UPLOAD_TYPES = ["csv", "xlsx", "xls", "json"]


st.set_page_config(
    page_title="DataNarrate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetricValue"] { font-size: 1.75rem; }
    .dn-hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, #111827 0%, #1d4ed8 52%, #06b6d4 100%);
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
    }
    .dn-hero h1 { margin: 0; font-size: 2.65rem; letter-spacing: -0.04em; }
    .dn-hero p { margin: .45rem 0 0 0; font-size: 1.05rem; opacity: .92; }
    .dn-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        background: rgba(248, 250, 252, 0.72);
        margin-bottom: 1rem;
    }
    .dn-small { color: #64748b; font-size: .9rem; }
    .dn-pill {
        display: inline-block;
        border-radius: 999px;
        padding: .18rem .62rem;
        margin: .1rem .15rem .1rem 0;
        background: #e0f2fe;
        color: #075985;
        font-size: .82rem;
        font-weight: 600;
    }
    .dn-warning { color: #92400e; background: #fffbeb; border: 1px solid #fde68a; padding: .8rem; border-radius: .8rem; }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Session and engine utilities
# -----------------------------


def run_async(coro):
    """Run an async engine endpoint from Streamlit's synchronous runtime."""
    return asyncio.run(coro)


def sanitize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim object columns and remove embedded newlines for cleaner previews/charts."""
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.replace("\n", " ").strip() if isinstance(x, str) else x)
    return df


def validate_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Validate a loaded DataFrame and normalize duplicate/blank column names."""
    if df.empty:
        raise ValueError(f"{dataset_name} did not contain any rows.")
    if len(df.columns) == 0:
        raise ValueError(f"{dataset_name} did not contain any columns.")

    normalized_columns = []
    seen: Dict[str, int] = {}
    for idx, col in enumerate(df.columns):
        base = str(col).strip() or f"unnamed_{idx + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        normalized_columns.append(base if count == 0 else f"{base}_{count + 1}")

    df = df.copy()
    df.columns = normalized_columns
    return df


def prepare_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Apply all load-time normalization before putting data into app state."""
    return sanitize_object_columns(validate_dataframe(df, dataset_name))


@st.cache_data(show_spinner=False)
def read_uploaded_dataset(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded dataset into a DataFrame."""
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(raw_bytes)

    if suffix == ".csv":
        df = pd.read_csv(buffer, engine="python", on_bad_lines="warn")
    elif suffix == ".xlsx":
        df = pd.read_excel(buffer, engine="openpyxl")
    elif suffix == ".xls":
        df = pd.read_excel(buffer, engine="xlrd")
    elif suffix == ".json":
        df = pd.read_json(buffer)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return prepare_dataframe(df, filename)


@st.cache_data(show_spinner=False)
def read_sample_dataset() -> pd.DataFrame:
    """Load the bundled sample dataset."""
    if engine.SAMPLE_DATASET_XLSX.exists():
        return prepare_dataframe(pd.read_excel(engine.SAMPLE_DATASET_XLSX, engine="openpyxl"), engine.SAMPLE_DATASET_XLSX.name)
    if engine.SAMPLE_DATASET_CSV.exists():
        return prepare_dataframe(pd.read_csv(engine.SAMPLE_DATASET_CSV, engine="python", on_bad_lines="warn"), engine.SAMPLE_DATASET_CSV.name)
    raise FileNotFoundError("Sample dataset not found")


def init_session_state() -> None:
    defaults = {
        "dataset_name": None,
        "df": None,
        "df_cleaned": None,
        "profile": None,
        "cleaning_log": [],
        "viz_plan": [],
        "chart_stats": {},
        "target_column": None,
        "target_candidates": [],
        "model_results": None,
        "has_trained_model": False,
        "summary": None,
        "intent": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def sync_engine_state() -> None:
    """Copy Streamlit state into the reusable analytics engine state."""
    engine.app_state.update({
        "df": st.session_state.df,
        "df_cleaned": st.session_state.df_cleaned,
        "dataset_name": st.session_state.dataset_name,
        "dataset_path": None,
        "profile": st.session_state.profile,
        "cleaning_log": st.session_state.cleaning_log,
        "viz_plan": st.session_state.viz_plan,
        "chart_stats": st.session_state.chart_stats,
        "column_name_mapping": engine.app_state.get("column_name_mapping", {}),
        "target_column": st.session_state.target_column,
        "target_candidates": st.session_state.target_candidates,
        "model": engine.app_state.get("model"),
        "model_results": st.session_state.model_results,
        "has_trained_model": st.session_state.has_trained_model,
    })


def pull_engine_state() -> None:
    """Persist engine outputs back into Streamlit session state."""
    for key in [
        "df_cleaned",
        "profile",
        "cleaning_log",
        "viz_plan",
        "chart_stats",
        "target_column",
        "target_candidates",
        "model_results",
        "has_trained_model",
    ]:
        st.session_state[key] = engine.app_state.get(key)


def reset_analysis(df: pd.DataFrame, dataset_name: str) -> None:
    st.session_state.dataset_name = dataset_name
    st.session_state.df = df
    st.session_state.df_cleaned = None
    st.session_state.profile = None
    st.session_state.cleaning_log = []
    st.session_state.viz_plan = []
    st.session_state.chart_stats = {}
    st.session_state.target_column = None
    st.session_state.target_candidates = []
    st.session_state.model_results = None
    st.session_state.has_trained_model = False
    st.session_state.summary = None
    engine.app_state.update({
        "df": df,
        "df_cleaned": None,
        "dataset_name": dataset_name,
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
    })


def active_df() -> Optional[pd.DataFrame]:
    return st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df


def ensure_profile() -> Optional[Dict[str, Any]]:
    df = active_df()
    if df is None:
        return None
    if st.session_state.profile is None:
        st.session_state.profile = engine.profile_dataframe(df)
    sync_engine_state()
    return st.session_state.profile


def response_data(response: Dict[str, Any]) -> Optional[Any]:
    if not response.get("success"):
        st.error(response.get("error", "Something went wrong."))
        return None
    return response.get("data")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def charts_zip_bytes() -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for chart_path in sorted(engine.CHARTS_DIR.glob("*.png")):
            archive.write(chart_path, chart_path.name)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# -----------------------------
# UI fragments
# -----------------------------


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="dn-hero">
            <h1>📊 {APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("1. Load data")
        source = st.radio("Choose a source", ["Upload file", "Use sample dataset"], label_visibility="collapsed")

        if source == "Upload file":
            upload = st.file_uploader("Upload CSV, Excel, or JSON", type=SUPPORTED_UPLOAD_TYPES)
            if upload is not None:
                raw = upload.getvalue()
                fingerprint = (upload.name, len(raw))
                if st.session_state.get("upload_fingerprint") != fingerprint:
                    try:
                        with st.spinner("Reading dataset..."):
                            df = read_uploaded_dataset(raw, upload.name)
                        st.session_state.upload_fingerprint = fingerprint
                        reset_analysis(df, upload.name)
                        st.success(f"Loaded {upload.name}")
                    except Exception as exc:
                        st.error(f"Could not read {upload.name}: {exc}")
        else:
            if st.button("Load bundled AFCON sample", use_container_width=True):
                try:
                    with st.spinner("Loading sample dataset..."):
                        df = read_sample_dataset()
                    reset_analysis(df, "AFCON 2025-2026 sample")
                    st.success("Sample dataset loaded")
                except Exception as exc:
                    st.error(f"Could not load the bundled sample: {exc}")

        st.divider()
        st.header("2. Describe your goal")
        st.session_state.intent = st.text_area(
            "Analysis intent",
            value=st.session_state.intent,
            placeholder="Example: Which factors are associated with match outcomes, and can we predict the winner?",
            height=120,
        )
        st.caption("The app uses this to prioritize charts, summaries, and target suggestions.")

        st.divider()
        st.header("Environment")
        llm_ready = bool(engine.GROQ_API_KEYS)
        st.write("LLM summaries:", "✅ enabled" if llm_ready else "⚠️ fallback mode")
        st.write("Primary model:", engine.PRIMARY_MODEL)
        if st.session_state.dataset_name:
            st.write("Dataset:", st.session_state.dataset_name)
            st.write("Rows:", f"{len(active_df()):,}" if active_df() is not None else "—")


def render_dataset_overview(df: pd.DataFrame) -> None:
    profile = ensure_profile()
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing values", f"{missing:,}")
    c4.metric("Duplicate rows", f"{duplicates:,}")

    if profile:
        st.markdown(
            f"<div class='dn-card'><strong>Profile summary</strong><br><span class='dn-small'>{profile.get('summary_text', '')}</span></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Preview data", expanded=True):
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)


def render_quality_tab() -> None:
    profile = ensure_profile()
    if not profile:
        st.info("Load a dataset to see quality profiling.")
        return

    st.subheader("Data quality profile")
    col_profiles = pd.DataFrame(profile.get("column_profiles", []))
    if not col_profiles.empty:
        display_cols = [
            "name", "column_type", "missing_pct", "unique_count", "is_constant", "outlier_pct", "high_cardinality",
        ]
        display_cols = [col for col in display_cols if col in col_profiles.columns]
        st.dataframe(col_profiles[display_cols], use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Numeric columns**")
    c1.markdown(" ".join([f"<span class='dn-pill'>{col}</span>" for col in profile.get("numeric_cols", [])]) or "None", unsafe_allow_html=True)
    c2.markdown("**Categorical columns**")
    c2.markdown(" ".join([f"<span class='dn-pill'>{col}</span>" for col in profile.get("categorical_cols", [])]) or "None", unsafe_allow_html=True)
    c3.markdown("**Datetime columns**")
    c3.markdown(" ".join([f"<span class='dn-pill'>{col}</span>" for col in profile.get("datetime_cols", [])]) or "None", unsafe_allow_html=True)


def render_cleaning_tab() -> None:
    if st.session_state.df is None:
        st.info("Load a dataset before cleaning.")
        return

    st.subheader("Clean data")
    st.write("Safe cleaning standardizes missing tokens, trims text, normalizes column names, and removes exact duplicate rows. Recommended cleaning can also impute missing values and handle obvious quality issues.")

    c1, c2 = st.columns([1, 1])
    with c1:
        safe_clicked = st.button("Apply safe cleaning", type="primary", use_container_width=True)
    with c2:
        recommended_clicked = st.button("Apply recommended cleaning", use_container_width=True)

    if safe_clicked or recommended_clicked:
        sync_engine_state()
        with st.spinner("Cleaning dataset..."):
            response = run_async(engine.apply_cleaning(engine.CleanRequest(apply_recommended=recommended_clicked)))
        data = response_data(response)
        if data:
            pull_engine_state()
            st.success(f"Cleaned dataset: {data['n_rows']:,} rows × {data['n_cols']:,} columns")

    log = st.session_state.cleaning_log or []
    if log:
        st.markdown("#### Cleaning log")
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

    df = active_df()
    if df is not None:
        st.download_button(
            "Download current dataset as CSV",
            data=dataframe_to_csv_bytes(df),
            file_name=f"datanarrate_{Path(st.session_state.dataset_name or 'dataset').stem}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_visuals_tab() -> None:
    if active_df() is None:
        st.info("Load a dataset before generating visuals.")
        return

    st.subheader("Intent-aware visualizations")
    st.write("Generate seven de-duplicated charts. If a Groq key is configured, the plan is tailored by the LLM; otherwise DataNarrate falls back to strong rule-based charts.")

    c1, c2 = st.columns([1, 1])
    with c1:
        plan_clicked = st.button("Create chart plan", use_container_width=True)
    with c2:
        charts_clicked = st.button("Generate charts", type="primary", use_container_width=True)

    intent = st.session_state.intent or "Explore the most important patterns in this dataset."

    if plan_clicked:
        ensure_profile()
        sync_engine_state()
        with st.spinner("Planning charts..."):
            response = run_async(engine.generate_plan(engine.PlanRequest(intent=intent, allow_sample_rows=False)))
        data = response_data(response)
        if data:
            pull_engine_state()
            st.success("Chart plan ready")
            st.caption(data.get("reasoning", ""))

    if charts_clicked:
        ensure_profile()
        if not st.session_state.viz_plan:
            sync_engine_state()
            response = run_async(engine.generate_plan(engine.PlanRequest(intent=intent, allow_sample_rows=False)))
            response_data(response)
            pull_engine_state()
        sync_engine_state()
        with st.spinner("Generating charts..."):
            response = run_async(engine.generate_charts(engine.ChartsRequest(intent=intent)))
        data = response_data(response)
        if data:
            pull_engine_state()
            st.success(f"Generated {len(data.get('charts', []))} charts")

    if st.session_state.viz_plan:
        with st.expander("Chart plan", expanded=False):
            st.json(st.session_state.viz_plan)

    chart_stats = st.session_state.chart_stats or {}
    if chart_stats:
        st.markdown("#### Charts")
        sorted_charts = sorted(chart_stats.values(), key=lambda item: item.get("index", 0))
        for left, right in zip(sorted_charts[0::2], sorted_charts[1::2]):
            cols = st.columns(2)
            render_chart_card(cols[0], left)
            render_chart_card(cols[1], right)
        if len(sorted_charts) % 2:
            render_chart_card(st.container(), sorted_charts[-1])

        st.download_button(
            "Download all charts as ZIP",
            data=charts_zip_bytes(),
            file_name=f"datanarrate_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
        )


def render_chart_card(container, chart: Dict[str, Any]) -> None:
    filename = chart.get("filename")
    path = engine.CHARTS_DIR / filename if filename else None
    with container:
        if path and path.exists():
            st.image(str(path), use_container_width=True)
        title = chart.get("title") or chart.get("chart_type") or chart.get("type") or "Chart"
        st.markdown(f"**{title}**")
        explanation = chart.get("explanation") or chart.get("description") or chart.get("insight")
        if explanation:
            st.caption(explanation)


def infer_default_task_type(df: pd.DataFrame, target: str) -> str:
    series = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    unique_count = series.nunique()
    unique_ratio = unique_count / max(len(series), 1)
    return "classification" if unique_count <= 10 or unique_ratio < 0.05 else "regression"


def render_model_tab() -> None:
    df = active_df()
    if df is None:
        st.info("Load a dataset before training a model.")
        return

    st.subheader("Baseline machine learning")
    st.markdown(
        "<div class='dn-warning'>Baseline models are for pattern discovery, not production decisions. Review leakage, sampling, and domain assumptions before acting on results.</div>",
        unsafe_allow_html=True,
    )

    intent = st.session_state.intent or "Suggest the best prediction target for this dataset."
    if st.button("Suggest target from intent", use_container_width=True):
        ensure_profile()
        sync_engine_state()
        with st.spinner("Reviewing columns and intent..."):
            response = run_async(engine.infer_target_column(engine.TargetInferenceRequest(
                intent=intent,
                allow_sample_rows=False,
            )))
        data = response_data(response)
        if data:
            pull_engine_state()
            st.success(data.get("assumption_note", "Target suggestions ready"))

    candidates = st.session_state.target_candidates or []
    if candidates:
        with st.expander("Target suggestions", expanded=True):
            st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)

    columns = list(df.columns)
    default_target = st.session_state.target_column or (candidates[0]["column"] if candidates else columns[0])
    target_index = columns.index(default_target) if default_target in columns else 0
    target = st.selectbox("Target column", columns, index=target_index)
    st.session_state.target_column = target
    task_type = st.radio("Task type", ["classification", "regression"], index=0 if infer_default_task_type(df, target) == "classification" else 1, horizontal=True)
    test_size = st.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    if st.button("Train baseline model", type="primary", use_container_width=True):
        st.session_state.model_results = None
        st.session_state.has_trained_model = False
        engine.app_state["model"] = None
        engine.app_state["model_results"] = None
        engine.app_state["has_trained_model"] = False
        ensure_profile()
        sync_engine_state()
        with st.spinner("Training model..."):
            response = run_async(engine.train_model(engine.ModelTrainRequest(
                target_column=target,
                task_type=task_type,
                test_size=float(test_size),
            )))
        data = response_data(response)
        if data:
            pull_engine_state()
            st.success("Baseline model trained")

    results = st.session_state.model_results
    if results:
        st.markdown("#### Model results")
        metrics = results.get("metrics", {})
        if results.get("task_type") == "classification":
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            c2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.1%}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{metrics.get('rmse', 0):,.3f}")
            c2.metric("MAE", f"{metrics.get('mae', 0):,.3f}")
            c3.metric("R²", f"{metrics.get('r2', 0):.3f}")

        summary = results.get("model_summary", {})
        st.json({k: v for k, v in summary.items() if k != "feature_importance"}, expanded=False)

        importance = summary.get("feature_importance", [])
        if importance:
            st.markdown("#### Top model features")
            st.dataframe(pd.DataFrame(importance).head(20), use_container_width=True, hide_index=True)

        artifact = results.get("artifacts", {})
        artifact_name = artifact.get("filename")
        if artifact_name:
            artifact_path = engine.MODELING_DIR / artifact_name
            if artifact_path.exists():
                st.image(str(artifact_path), use_container_width=True)


def render_summary_tab() -> None:
    if active_df() is None:
        st.info("Load a dataset before generating a summary.")
        return

    st.subheader("Executive summary")
    st.write("Generate a plain-English answer to your original question using the profile, cleaning actions, charts, and model results.")

    if st.button("Generate final summary", type="primary", use_container_width=True):
        ensure_profile()
        sync_engine_state()
        with st.spinner("Writing summary..."):
            response = run_async(engine.generate_final_summary(engine.FinalSummaryRequest(
                intent=st.session_state.intent or "Summarize the most important findings.",
                allow_sample_rows=False,
            )))
        data = response_data(response)
        if data:
            st.session_state.summary = data.get("summary")
            st.success("Summary generated")

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        st.download_button(
            "Download summary",
            data=st.session_state.summary.encode("utf-8"),
            file_name="datanarrate_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


# -----------------------------
# Main app
# -----------------------------


def main() -> None:
    init_session_state()
    render_sidebar()
    render_hero()

    df = active_df()
    if df is None:
        st.info("Upload a dataset or load the bundled sample from the sidebar to begin.")
        st.markdown(
            """
            ### What this Streamlit version does
            - Profiles data quality and column types automatically.
            - Applies safe or recommended cleaning with an auditable log.
            - Creates seven intent-aware charts with fallback planning when LLM access is unavailable.
            - Trains a baseline classification or regression model with model diagnostics.
            - Produces an executive summary and downloadable outputs.
            """
        )
        return

    render_dataset_overview(df)

    tabs = st.tabs(["Quality", "Cleaning", "Visuals", "Model", "Summary"])
    with tabs[0]:
        render_quality_tab()
    with tabs[1]:
        render_cleaning_tab()
    with tabs[2]:
        render_visuals_tab()
    with tabs[3]:
        render_model_tab()
    with tabs[4]:
        render_summary_tab()


if __name__ == "__main__":
    main()
