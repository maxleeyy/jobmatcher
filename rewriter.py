import os
import streamlit as st
from typing import Literal

def _get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

def rewrite_resume(resume_text: str,
                   job_desc: str,
                   provider: Literal['openai']='openai',
                   model: str='gpt-4o-mini',
                   template: str=None,
                   system_prompt: str=None,
                   temperature: float=0.2) -> str:
    if provider != 'openai':
        return "[ERROR] Only provider='openai' is implemented."
    return _rewrite_openai(resume_text, job_desc, model, template, system_prompt, temperature)

def _rewrite_openai(resume_text: str, job_desc: str, model: str,
                    template: str, system_prompt: str, temperature: float) -> str:
    from openai import OpenAI
    api_key = _get_openai_key()
    if not api_key:
        return "[ERROR] OPENAI_API_KEY not set. Add it in Streamlit → Settings → Secrets."
    client = OpenAI(api_key=api_key)

    if not system_prompt:
        system_prompt = ("You are an expert HR resume coach. Preserve factual accuracy. "
                         "Improve clarity and ATS alignment.")
    if not template:
        template = ("Rewrite to match the job description without adding new facts.\n\n"
                    "JD:\n{job_desc}\n\nRESUME:\n{resume}")

    user_prompt = template.format(job_desc=job_desc, resume=resume_text)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
