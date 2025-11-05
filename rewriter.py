import os
import streamlit as st
from typing import Literal

def _get_openai_key() -> str:
    # Try env var first, then Streamlit secrets
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

def rewrite_resume(resume_text: str, job_desc: str,
                   provider: Literal['openai']='openai',
                   model: str='gpt-4o-mini') -> str:
    if provider == 'openai':
        return _rewrite_openai(resume_text, job_desc, model)
    raise ValueError('Only provider="openai" implemented in this template.')

def _rewrite_openai(resume_text: str, job_desc: str, model: str) -> str:
    from openai import OpenAI
    api_key = _get_openai_key()
    if not api_key:
        return "[ERROR] OPENAI_API_KEY not set. Add it in Streamlit → Settings → Secrets."

    client = OpenAI(api_key=api_key)

    system = (
        "You are an expert HR resume coach. Preserve factual accuracy; "
        "do not invent employers, dates, titles, or metrics. Rephrase for clarity and ATS."
    )
    user = f"""
Rewrite the following resume to better align with the job description.
- Keep facts truthful; do NOT add experience or skills not present.
- Emphasize relevant skills, tools, and achievements.
- Use concise bullets; keep under ~2 pages as text.

JOB DESCRIPTION:
{job_desc}

RESUME (SOURCE TRUTH):
{resume_text}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

