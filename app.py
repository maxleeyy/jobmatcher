import os, io
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# ------------------ UI SETUP ------------------
st.set_page_config(page_title="AI Job Matcher + CV Optimizer", page_icon="🎯", layout="wide")
st.title("🤖 AI-Powered Job Matcher & CV Optimizer")
st.write("Upload or paste your resume and job description to get a match score, keyword alignment, an AI-optimized resume, and evaluation metrics.")

# ------------------ PROMPT TEMPLATES ------------------
BASE_SYSTEM = (
    "You are an expert HR resume coach. Preserve factual accuracy — "
    "do not invent employers, dates, titles, or metrics. Improve clarity, "
    "ATS compatibility, and relevance to the target job."
)
TEMPLATES = {
    "Full Resume Rewrite": """Rewrite the resume to better align with the job description.
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
    "Summary Only": """Rewrite ONLY the resume SUMMARY to fit the job description.
- Keep facts truthful; do not add non-existent skills.
- 3–5 lines, skills-forward, include 1 metric if present.

JOB DESCRIPTION:
{job_desc}

RESUME SUMMARY:
{resume}
""",
    "Skills Alignment": """Produce a revised SKILLS section aligned to the job description.
- Include only skills actually present in the resume.
- Group by categories (Programming, Data, Cloud, Tools, etc.)
- Keep it compact and ATS-friendly.

JOB DESCRIPTION:
{job_desc}

RESUME SKILLS:
{resume}
""",
    "STAR Bullets (Experience)": """Rewrite the EXPERIENCE bullets using the STAR pattern (Situation-Task-Action-Result)
for RELEVANT roles only. Do not fabricate metrics.

JOB DESCRIPTION:
{job_desc}

RESUME EXPERIENCE:
{resume}
""",
    "CN/EN Bilingual Summary": """Write a bilingual summary (Chinese + English), 2–3 lines each, aligned to the job description.
Keep strictly to resume facts.

JOB DESCRIPTION:
{job_desc}

RESUME SUMMARY:
{resume}
""",
}

# ------------------ FILE READERS ------------------
def read_any(uploaded_file) -> str:
    """Return plain text from TXT/PDF/DOCX using pypdf + python-docx."""
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
            if blank <= 1: out.append("")
        else:
            blank = 0; out.append(l)
    return "\n".join(out).strip()

# ------------------ MATCHING ------------------
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
    def toks(s): return re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{1,}", s.lower())
    stop = set("the a an and or of to for with on in from by as at is are be been was were you your our their they them it its this that".split())
    rset = set([t for t in toks(resume_text) if t not in stop and len(t) >= 2])
    jtok = [t for t in toks(jd_text) if t not in stop and len(t) >= 2]
    freq = collections.Counter(jtok)
    missing = [w for w,_ in freq.most_common(60) if w not in rset][:20]
    present = [w for w,_ in freq.most_common(60) if w in rset][:20]
    return missing, present

# ------------------ OPENAI REWRITER ------------------
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

# ------------------ INPUTS ------------------
st.markdown("### 📄 Resume Input")
col_r1, col_r2 = st.columns(2)
with col_r1:
    up_resume = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf","docx","txt"], key="resume")
with col_r2:
    up_jd = st.file_uploader("Upload Job Description (PDF / DOCX / TXT)", type=["pdf","docx","txt"], key="jd")

resume_text = read_any(up_resume) if up_resume else ""
jd_text    = read_any(up_jd) if up_jd else ""
if not resume_text:
    resume_text = st.text_area("Or paste your Resume here", height=240, key="resume_text_area")
if not jd_text:
    jd_text = st.text_area("Or paste Job Description here", height=200, key="jd_text_area")
resume_text = clean_text(resume_text); jd_text = clean_text(jd_text)

# ------------------ SETTINGS ------------------
st.markdown("### ⚙️ Settings")
c1, c2, c3 = st.columns(3)
with c1:
    tpl_name = st.selectbox("Rewrite Template", list(TEMPLATES.keys()), index=0)
with c2:
    model_choice = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
with c3:
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1)

st.markdown("---")

# ------------------ MAIN RUN ------------------
if st.button("🚀 Analyze & Optimize", use_container_width=True):
    if not resume_text or not jd_text:
        st.error("Please provide both the resume and the job description (upload or paste).")
    else:
        with st.spinner("Calculating Job Match Score..."):
            score = compute_match_score(resume_text, jd_text)
            missing, present = keyword_gap(resume_text, jd_text)
        st.success(f"🎯 Job Match Score: **{score}%**")

        with st.expander("🔎 Keyword Gap Analysis", expanded=True):
            a, b = st.columns(2)
            with a:
                st.markdown("**Missing keywords (consider addressing):**")
                st.write(", ".join(missing) if missing else "None — great alignment.")
            with b:
                st.markdown("**Already present:**")
                st.write(", ".join(present) if present else "—")

        st.markdown("---")
        with st.spinner("Rewriting resume (facts preserved)..."):
            optimized = rewrite_resume(
                resume_text, jd_text,
                TEMPLATES[tpl_name], BASE_SYSTEM,
                model_choice, temperature
            )

        st.subheader("🧠 Optimized Resume")
        st.text_area("", optimized, height=420)
        st.download_button("💾 Download Optimized Resume (.txt)", optimized, file_name="optimized_resume.txt", use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧪 A/B Model Comparison (optional)")
        if st.checkbox("Compare two models side-by-side"):
            m2 = st.selectbox("Second model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=2, key="m2")
            colx, coly = st.columns(2)
            with colx:
                st.write(f"**Model: {model_choice}**")
                out1 = rewrite_resume(resume_text, jd_text, TEMPLATES[tpl_name], BASE_SYSTEM, model_choice, temperature)
                st.text_area("", out1, height=320)
            with coly:
                st.write(f"**Model: {m2}**")
                out2 = rewrite_resume(resume_text, jd_text, TEMPLATES[tpl_name], BASE_SYSTEM, m2, temperature)
                st.text_area("", out2, height=320)

# ------------------ INLINE EVALUATION ------------------
import pandas as pd
import numpy as np
from pathlib import Path

def _read_text_for_eval(p: str) -> str:
    return Path(p).read_text(encoding="utf-8", errors="ignore")

def _precision_at_k(labels_sorted, k=3):
    topk = labels_sorted[:k]
    positives = sum(1 for y in topk if y == 2)   # "Good"
    return positives / k

def _mrr(relevance_list):
    for i, rel in enumerate(relevance_list, start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0

with st.expander("📊 Evaluation (rank accuracy)"):
    st.write("Loads `eval_pairs.csv` from repo root and computes Precision@3 + MRR using the same match scorer.")
    csv_path = Path("eval_pairs.csv")
    if not csv_path.exists():
        st.info("`eval_pairs.csv` not found. Create it and referenced text files, then reload.")
    else:
        if st.button("Run evaluation now"):
            try:
                df = pd.read_csv(csv_path)
                results, p3s, mrrs = [], [], []
                for jd_path, group in df.groupby("jd_path"):
                    jd_text_local = _read_text_for_eval(jd_path)
                    scored = []
                    for _, row in group.iterrows():
                        r_text_local = _read_text_for_eval(row["resume_path"])
                        s = compute_match_score(r_text_local, jd_text_local)
                        scored.append((row["resume_path"], s, int(row["label"])))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    y_sorted = [lab for _, _, lab in scored]
                    rel_bin = [1 if y == 2 else 0 for y in y_sorted]
                    p3s.append(_precision_at_k(y_sorted, k=3))
                    mrrs.append(_mrr(rel_bin))
                    results.extend([(jd_path, rp, round(s, 2), {0:"Poor",1:"Medium",2:"Good"}[lab])
                                    for rp, s, lab in scored])

                out = pd.DataFrame(results, columns=["JD","Resume","PredictedScore","TrueLabel"])
                st.dataframe(out, use_container_width=True)
                st.success(f"Precision@3 (avg): {np.mean(p3s):.3f}  ·  MRR (avg): {np.mean(mrrs):.3f}")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.info("Check file paths in `eval_pairs.csv` and ensure they exist under your repo.")
                
st.markdown("---")
st.caption("Privacy: processed in memory only. Optimizes presentation; does not alter qualifications.")
