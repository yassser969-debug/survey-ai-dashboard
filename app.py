import os
import io
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from openai import OpenAI

st.set_page_config(page_title="AI Survey Dashboard", layout="wide")


# -----------------------------
# Helper functions
# -----------------------------

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\xa0", "", regex=False).str.strip()
    return df


def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    data = df[column].fillna("Missing").astype(str).str.strip()
    counts = data.value_counts(dropna=False)
    total = len(data)

    return pd.DataFrame({
        "Response": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / total * 100).round(1) if total else 0,
    })


def cronbach_alpha(data: pd.DataFrame):
    numeric_data = data.apply(pd.to_numeric, errors="coerce").dropna()
    k = numeric_data.shape[1]
    n = len(numeric_data)

    if k < 2 or n < 2:
        return None, n, k

    item_variances = numeric_data.var(axis=0, ddof=1)
    total_variance = numeric_data.sum(axis=1).var(ddof=1)

    if total_variance == 0 or pd.isna(total_variance):
        return None, n, k

    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha), n, k


def alpha_interpretation(alpha):
    if alpha is None:
        return "Not enough valid data"
    if alpha >= 0.90:
        return "Excellent reliability"
    if alpha >= 0.80:
        return "Good reliability"
    if alpha >= 0.70:
        return "Acceptable reliability"
    if alpha >= 0.60:
        return "Questionable reliability"
    return "Poor reliability"


def build_ai_prompt(task, df, question=None, summary=None, alpha=None, alpha_label=None, extra_context=""):
    sample_size = len(df)

    if summary is not None:
        summary_text = summary.to_string(index=False)
    else:
        summary_text = "No summary table provided."

    return f"""
You are an academic survey data analyst.

Task:
{task}

Rules:
- Use only the numbers provided below.
- Do not invent data, statistics, tests, or conclusions.
- Mention limitations clearly, especially small sample size.
- Write in a formal academic style.
- Keep the interpretation concise and useful for a research report.
- If the evidence is weak, say it is weak.

Sample size:
{sample_size}

Question:
{question}

Results table:
{summary_text}

Cronbach's Alpha:
{alpha}

Alpha interpretation:
{alpha_label}

Additional context:
{extra_context}
"""


def get_openai_client(api_key):
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_ai_text(api_key, prompt, model):
    client = get_openai_client(api_key)

    if client is None:
        return "OpenAI API key is missing. Add it in the sidebar for local testing or use Streamlit Secrets when deployed."

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output_text
    except Exception as e:
        return f"AI generation failed: {e}"


def safe_excel_sheet_name(name: str, existing_names=None) -> str:
    if existing_names is None:
        existing_names = set()

    name = str(name)
    name = re.sub(r'[\[\]\:\*\?\/\\]', "_", name)
    name = name[:31].strip()

    if not name:
        name = "Sheet"

    original_name = name
    counter = 1

    while name in existing_names:
        suffix = f"_{counter}"
        name = original_name[:31 - len(suffix)] + suffix
        counter += 1

    existing_names.add(name)
    return name


def excel_download(all_tables: dict) -> bytes:
    output = io.BytesIO()
    used_sheet_names = set()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, table in all_tables.items():
            safe_name = safe_excel_sheet_name(name, used_sheet_names)
            table.to_excel(writer, sheet_name=safe_name, index=False)

    return output.getvalue()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Settings")

api_key_from_secrets = ""
try:
    api_key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    api_key_from_secrets = ""

api_key_input = st.sidebar.text_input(
    "OpenAI API Key (local testing only)",
    type="password",
    help="For deployed apps, use Streamlit Secrets instead of typing the key here.",
)

api_key = api_key_from_secrets or api_key_input or os.getenv("OPENAI_API_KEY", "")

try:
    default_model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
except Exception:
    default_model = "gpt-4o-mini"

model = st.sidebar.text_input("OpenAI model", value=default_model)

st.sidebar.warning(
    "Do not upload sensitive personal data unless you have permission and proper data protection controls."
)


# -----------------------------
# Main app
# -----------------------------

st.title("AI Survey Dashboard")
st.caption(
    "Upload an Excel questionnaire file, explore results, filter responses, run reliability analysis, and generate AI-assisted academic interpretation."
)

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is None:
    st.info("Upload an Excel file to start.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
    df = clean_columns(df)
except Exception as e:
    st.error(f"Could not read the Excel file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

columns = list(df.columns)

st.success(f"File loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns.")

with st.expander("Data preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Column names", expanded=False):
    st.write(columns)


# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Single Question",
    "Filtered Analysis",
    "Crosstab",
    "Cronbach Alpha",
    "AI Report",
])


# -----------------------------
# Overview
# -----------------------------

with tab1:
    st.header("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Responses", df.shape[0])
    c2.metric("Variables", df.shape[1])
    c3.metric("Missing cells", int(df.isna().sum().sum()))

    st.subheader("Missing values by variable")

    missing_table = pd.DataFrame({
        "Variable": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing Percentage": (df.isna().sum().values / len(df) * 100).round(1),
    }).sort_values("Missing Count", ascending=False)

    st.dataframe(missing_table, use_container_width=True)


# -----------------------------
# Single Question
# -----------------------------

with tab2:
    st.header("Single Question Analysis")

    question = st.selectbox("Select question", columns, key="single_question")
    chart_type = st.radio("Chart type", ["Bar", "Pie"], horizontal=True, key="single_chart")

    summary = frequency_table(df, question)
    st.dataframe(summary, use_container_width=True)

    if chart_type == "Bar":
        fig = px.bar(summary, x="Response", y="Count", text="Count", title=question)
    else:
        fig = px.pie(summary, names="Response", values="Count", title=question)

    st.plotly_chart(fig, use_container_width=True)

    if st.button("Generate AI interpretation", key="ai_single"):
        prompt = build_ai_prompt(
            task="Interpret this single questionnaire item.",
            df=df,
            question=question,
            summary=summary,
        )

        with st.spinner("Generating AI interpretation..."):
            st.write(generate_ai_text(api_key, prompt, model))


# -----------------------------
# Filtered Analysis
# -----------------------------

with tab3:
    st.header("Filtered Analysis")

    filter_col = st.selectbox("Filter by", columns, key="filter_col")
    filter_values = ["All"] + sorted(df[filter_col].dropna().astype(str).unique().tolist())
    filter_value = st.selectbox("Filter value", filter_values, key="filter_value")

    analysis_col = st.selectbox("Analyze question", columns, key="analysis_col")
    filtered_chart = st.radio("Chart type", ["Bar", "Pie"], horizontal=True, key="filtered_chart")

    if filter_value == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df[filter_col].astype(str) == str(filter_value)]

    st.write(f"Filtered responses: **{len(filtered_df)}**")

    if len(filtered_df) == 0:
        st.warning("No data available for this filter.")
    else:
        filtered_summary = frequency_table(filtered_df, analysis_col)
        st.dataframe(filtered_summary, use_container_width=True)

        if filtered_chart == "Bar":
            fig = px.bar(
                filtered_summary,
                x="Response",
                y="Count",
                text="Count",
                title=f"{analysis_col} | {filter_col}: {filter_value}",
            )
        else:
            fig = px.pie(
                filtered_summary,
                names="Response",
                values="Count",
                title=f"{analysis_col} | {filter_col}: {filter_value}",
            )

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Generate AI interpretation", key="ai_filtered"):
            prompt = build_ai_prompt(
                task="Interpret this filtered questionnaire result.",
                df=filtered_df,
                question=analysis_col,
                summary=filtered_summary,
                extra_context=f"Filter applied: {filter_col} = {filter_value}",
            )

            with st.spinner("Generating AI interpretation..."):
                st.write(generate_ai_text(api_key, prompt, model))


# -----------------------------
# Crosstab
# -----------------------------

with tab4:
    st.header("Crosstab Analysis")

    row_var = st.selectbox("Rows", columns, key="row_var")
    col_var = st.selectbox("Columns", columns, key="col_var")
    normalize_by = st.selectbox("Percentage by", ["Rows", "Columns", "All"], key="normalize_by")

    if row_var == col_var:
        st.warning("Choose two different variables for crosstab analysis.")
    else:
        temp = df[[row_var, col_var]].copy()
        temp = temp.loc[:, ~temp.columns.duplicated()].copy()

        row_data = temp[row_var]
        col_data = temp[col_var]

        if isinstance(row_data, pd.DataFrame):
            row_data = row_data.iloc[:, 0]

        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]

        row_data = row_data.fillna("Missing").astype(str)
        col_data = col_data.fillna("Missing").astype(str)

        count_table = pd.crosstab(row_data, col_data)

        if normalize_by == "Rows":
            pct_table = pd.crosstab(row_data, col_data, normalize="index") * 100
        elif normalize_by == "Columns":
            pct_table = pd.crosstab(row_data, col_data, normalize="columns") * 100
        else:
            pct_table = pd.crosstab(row_data, col_data, normalize="all") * 100

        pct_table = pct_table.round(1)

        st.subheader("Count table")
        st.dataframe(count_table, use_container_width=True)

        st.subheader("Percentage table")
        st.dataframe(pct_table, use_container_width=True)

        count_plot = count_table.reset_index()

        index_col_name = count_plot.columns[0]

        count_plot = count_plot.melt(
            id_vars=index_col_name,
            var_name=col_var,
            value_name="Count",
        )

        fig = px.bar(
            count_plot,
            x=index_col_name,
            y="Count",
            color=col_var,
            barmode="group",
            title=f"{row_var} vs {col_var}",
        )

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Generate AI interpretation", key="ai_crosstab"):
            prompt = f"""
You are an academic survey data analyst.

Interpret the crosstab results below.

Rules:
- Use only the provided count and percentage tables.
- Do not invent any statistical significance.
- Mention that this is descriptive unless inferential testing is performed.
- Mention limitations if the sample is small.

Rows variable: {row_var}
Columns variable: {col_var}
Sample size: {len(df)}

Count table:
{count_table.to_string()}

Percentage table:
{pct_table.to_string()}
"""

            with st.spinner("Generating AI interpretation..."):
                st.write(generate_ai_text(api_key, prompt, model))


# -----------------------------
# Cronbach Alpha
# -----------------------------

with tab5:
    st.header("Cronbach Alpha Reliability Analysis")
    st.write("Select Likert-scale items that measure the same construct.")

    default_likert = [c for c in columns if any(str(c).startswith(f"Q{i}") for i in range(16, 24))]

    selected_likert = st.multiselect(
        "Likert items",
        columns,
        default=default_likert if default_likert else [],
    )

    if len(selected_likert) < 2:
        st.warning("Select at least two Likert-scale items.")
    else:
        alpha, valid_n, k = cronbach_alpha(df[selected_likert])
        alpha_label = alpha_interpretation(alpha)

        c1, c2, c3 = st.columns(3)
        c1.metric("Cronbach Alpha", "N/A" if alpha is None else round(alpha, 3))
        c2.metric("Valid responses", valid_n)
        c3.metric("Items", k)

        st.info(alpha_label)

        item_preview = df[selected_likert].apply(pd.to_numeric, errors="coerce")
        st.dataframe(item_preview.describe().T, use_container_width=True)

        if st.button("Generate AI reliability interpretation", key="ai_alpha"):
            prompt = build_ai_prompt(
                task="Interpret the Cronbach alpha reliability result.",
                df=df,
                question="Reliability analysis for selected Likert-scale items",
                summary=pd.DataFrame({"Selected items": selected_likert}),
                alpha=None if alpha is None else round(alpha, 3),
                alpha_label=alpha_label,
                extra_context="Cronbach's alpha is only meaningful if the selected items measure the same construct.",
            )

            with st.spinner("Generating AI interpretation..."):
                st.write(generate_ai_text(api_key, prompt, model))


# -----------------------------
# AI Report
# -----------------------------

with tab6:
    st.header("AI Report Generator")
    st.write("Generate a concise academic-style report based on selected variables.")

    report_questions = st.multiselect(
        "Select questions for the report",
        columns,
        default=columns[:min(5, len(columns))],
    )

    report_context = st.text_area(
        "Research context / notes",
        placeholder="Example: This survey examines lecturers' perceptions of assessment methods...",
    )

    if st.button("Generate AI report", key="ai_report"):
        if not report_questions:
            st.warning("Select at least one question.")
        else:
            report_sections = []

            for q in report_questions:
                q_summary = frequency_table(df, q)
                report_sections.append(f"Question: {q}\n{q_summary.to_string(index=False)}")

            report_prompt = f"""
You are an academic survey data analyst.

Write a concise results report based on the selected questionnaire items.

Rules:
- Use only the provided tables.
- Do not invent numbers.
- Do not claim statistical significance.
- Mention limitations, especially small sample size.
- Structure the answer with short headings.

Sample size: {len(df)}

Research context:
{report_context}

Question summaries:
{chr(10).join(report_sections)}
"""

            with st.spinner("Generating AI report..."):
                st.write(generate_ai_text(api_key, report_prompt, model))


# -----------------------------
# Download summaries
# -----------------------------

st.divider()
st.header("Download Summary Tables")

summary_tables = {col: frequency_table(df, col) for col in columns}
excel_bytes = excel_download(summary_tables)

st.download_button(
    label="Download summary Excel",
    data=excel_bytes,
    file_name="survey_summary_tables.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
