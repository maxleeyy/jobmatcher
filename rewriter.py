import os
from typing import Literal

def rewrite_resume(resume_text: str, job_desc: str, provider: Literal['openai']='openai', model: str='gpt-4o-mini') -> str:
    if provider == 'openai':
        return _rewrite_openai(resume_text, job_desc, model)
    raise ValueError('Only provider="openai" implemented in this template.')

def _rewrite_openai(resume_text: str, job_desc: str, model: str) -> str:
    from openai import OpenAI
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        return "[ERROR] OPENAI_API_KEY not set. Please configure your API key."
    client = OpenAI(api_key=api_key)
    system = (
        "You are an expert HR resume coach. You strictly preserve factual accuracy: "
        "never invent employers, dates, titles, or metrics. You may rephrase, reorder, "
        "and format for clarity and ATS compliance. Use strong action verbs and, when "
        "appropriate, STAR framing. Keep content concise."
    )
    user = f"""
Rewrite the following resume to better align with the job description.
- Keep facts truthful; do NOT add experience or skills that are not present.
- Emphasize relevant skills, tools, and achievements for this role.
- Improve formatting with bullet points where useful.
- Keep under 2 pages when rendered as text.

JOB DESCRIPTION:
{job_desc}

RESUME (SOURCE TRUTH):
{resume_text}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
