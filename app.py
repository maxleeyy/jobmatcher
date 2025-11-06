import os, io
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Job Matcher + CV Optimizer", page_icon="🎯", layout="wide")
st.title("🤖 AI-Powered Job Matcher & CV Optimizer")
st.write("Upload or paste your resume and job description to get a match score, keyword alignment, and an AI-optimized resume.")

# ----------------------------
# Prompt templates (built-in)
# ----------------------------
BASE_SYSTEM = (
    "You are an expert HR resume coach. Preserve factual accuracy — "
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
- 3–5 lines, skills-forward, include 1 metric if present.

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
Write a bilingual summary (Chinese + English), 2–3 lines each, aligned to the job description.
Keep strictly to resume facts.

JOB DESCRIPTION:
{job_desc}

RESUME SUMMARY:
{resume}
""",
}

# ----------------------------
# File readers (TXT / PDF / DOCX)
# ----------------------------
def read_any(uploaded_file) -> str:
    """Return plain text from uploaded TXT/PDF/DOCX (uses pypdf, python-docx)."""
    from pypdf import PdfReader
    from docx import Document

    name = (uploaded_file.name or "").lower()
    data = uploaded_file.read()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pass
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    # fallback try text decode
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def clean_text(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    lines = [x.strip() for x in t.split("\n")]
    out, blank = [], 0
    for l in lines:
        if l == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(l)
    return "\n".join(out).strip()

# ----------------------------
# Embeddings for Job Match Score
# ----------------------------
@st.cache_resource
def load_embed():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embed_model = load_embed()

def compute_match_score(resume_text: str, jd_text: str) -> float:
    r = embed_model.encode(resume_text, convert_to_tensor=True, normalize_embeddings=True)
    j = embed_model.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
    score = float(util.cos_sim(r, j).item())
    return round(max(0.0, min(1.0, score)) * 100, 2)

def keyword_gap(resume_text: str, jd_text: str):
    import re, collections
    def toks(s):
        return re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{1,}", s.lower())
    stop = set("the a an and or of to for with on in from by as at is are be been was were you your our their they them it its this that".split())
    rset = set([t for t in toks(resume_text) if t not in stop and len(t) >= 2])
    jtok = [t for t in toks(jd_text) if t not in stop and len(t) >= 2]
    freq = collections.Counter(jtok)
    missing = [w for w,_ in freq.most_common(60) if w not in rset][:20]
    present = [w for w,_ in freq.most_common(60) if w in rset][:20]
    return missing, present

# ----------------------------
# OpenAI client + rewrite
# ----------------------------
def get_openai_key():
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

def rewrite_resume(resume_text, jd_text, template, system_prompt, model_name, temperature=0.2):
    key = get_openai_key()
    if not key:
        return "[ERROR] OPENAI_API_KEY missing. Add it in Streamlit → Settings → Secrets."
    client = OpenAI(api_key=key)
    prompt = template.format(job_desc=jd_text, resume=resume_text)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error] {e}"

# ----------------------------
# Inputs: Upload OR Paste
# ----------------------------
st.markdown("### 📄 Resume Input")
col_r1, col_r2 = st.columns([1,1])
with col_r1:
    up_resume = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf","docx","txt"], key="resume")
with col_r2:
    up_jd = st.file_uploader("Upload Job Description (PDF / DOCX / TXT)", type=["pdf","docx","txt"], key="jd")

resume_text = read_any(up_resume) if up_resume else ""
jd_text = read_any(up_jd) if up_jd else ""

if not resume_text:
    resume_text = st.text_area("Or paste your Resume here", height=260, key="resume_text_area")
if not jd_text:
    jd_text = st.text_area("Or paste Job Description here", height=220, key="jd_text_area")

resume_text = clean_text(resume_text)
jd_text = clean_text(jd_text)

# ----------------------------
# Settings (template/model/temp)
# ----------------------------
