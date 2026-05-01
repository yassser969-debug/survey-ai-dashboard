# AI Survey Dashboard

A Streamlit web app for questionnaire analysis.

## Features

- Upload Excel questionnaire files
- Clean column names automatically
- Single-question frequency analysis
- Bar and pie charts
- Filtered analysis
- Crosstab analysis
- Cronbach Alpha reliability analysis
- AI-assisted academic interpretation using the OpenAI API
- Download summary tables as Excel

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud secrets

Add this in Streamlit Cloud > App settings > Secrets:

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-5.5"
```

Do not upload your real API key to GitHub.
