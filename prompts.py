# prompts.py
BASE_SYSTEM = (
    "You are an expert HR resume coach. Preserve factual accuracy—"
    "do not invent employers, dates, titles, or metrics. Improve clarity, "
    "ATS compatibility, and relevance to the target job."
)

TEMPLATES = {
    "Full Resume Rewrite": """\
Rewrite the resume to better align with the job description.
Rules:
- Keep facts truthful; DO NOT add new experience/skills.
- Emphasize relevant tools, skills, and achievements for this role.
- Use concise bullet points and impact verbs (STAR when possible).
- Keep under ~2 pages when rendered as text.

JOB DESCRIPTION:
{job_desc}

RESUME (SOURCE TRUTH):
{resume}
""",

    "Summary Only": """\
Rewrite ONLY the resume SUMMARY to fit the job description.
- Keep facts truthful; do not add non-existent skills.
- 3–5 lines, crisp, skills-forward, with 1 metric if present.

JOB DESCRIPTION:
{job_desc}

RESUME SUMMARY:
{resume}
""",

    "Skills Alignment": """\
Produce a revised SKILLS section aligned to the job description.
- Include only skills actually present in the resume.
- Group by categories (Programming, Data, Cloud, Tools, etc.)
- Keep it compact and ATS-friendly.

JOB DESCRIPTION:
{job_desc}

RESUME SKILLS:
{resume}
""",

    "STAR Bullets (Experience)": """\
Rewrite the EXPERIENCE bullets using the STAR pattern (Situation-Task-Action-Result)
for RELEVANT roles only. Do not fabricate metrics.

JOB DESCRIPTION:
{job_desc}

RESUME EXPERIENCE:
{resume}
""",

    "CN/EN Bilingual Summary": """\
Write a bilingual summary (Chinese + English), 2–3 lines each,
aligned to the job description. Keep strictly to resume facts.

JD:
{job_desc}

RESUME SUMMARY:
{resume}
""",
}
