import streamlit as st
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="AI Job Matcher + CV Optimizer", layout="wide")

# ========== PROMPT TEMPLATES ==========
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

# ========== LOAD MATCH MODEL ==========
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def compute_match_score(resume_text, job_desc):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_job = model.encode(job_desc, convert_to_tensor=True)
    score = util.cos_sim(emb_resume, emb_job).item()
    return round(score * 100, 2)

# ========== OPENAI REWRITE ==========
def get_openai_key():
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

def rewrite_resume(resume_text, job_desc, template, system_prompt, model_name, temperature=0.2):
    key = get_openai_key()
    if not key:
        return "[ERROR] Missing OPENAI_API_KEY. Add it in Streamlit → Settings → Secrets."
    client = OpenAI(api_key=key)

    user_prompt = template.format(job_desc=job_desc, resume=resume_text)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

# ========== STREAMLIT APP ==========
st.title("🤖 AI-Powered Job Matcher & CV Optimizer")
st.write("Upload or paste your resume and job description below to get your match score and optimized resume.")

resume_text = st.text_area("📄 Paste your Resume", height=300)
jd_text = st.text_area("💼 Paste Job Description", height=250)

# ===== Settings =====
st.markdown("### ⚙️ Settings")
colA, colB, colC = st.columns(3)
with colA:
    template_name = st.selectbox("Rewrite Template", list(TEMPLATES.keys()), index=0)
with colB:
    model_choice = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
with colC:
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1)

# ===== Analyze Button =====
if st.button("🚀 Analyze & Optimize", use_container_width=True):
    if not resume_text or not jd_text:
        st.error("Please paste both resume and job description.")
    else:
        with st.spinner("Calculating job match score..."):
            score = compute_match_score(resume_text, jd_text)
        st.success(f"🎯 Job Match Score: {score}%")

        st.markdown("---")
        with st.spinner("Rewriting resume with AI..."):
            optimized = rewrite_resume(
                resume_text,
                jd_text,
                TEMPLATES[template_name],
                BASE_SYSTEM,
                model_choice,
                temperature,
            )

        st.subheader("🧠 Optimized Resume")
        st.text_area("", optimized, height=400)
        st.download_button("💾 Download Optimized Resume", optimized, file_name="optimized_resume.txt")

        # ===== Optional A/B Testing =====
        st.markdown("---")
        st.markdown("### 🧪 A/B Model Comparison (optional)")
        ab = st.checkbox("Compare two models side-by-side")

        if ab:
            m1 = model_choice
            m2 = st.selectbox("Second model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=2, key="m2")
            colx, coly = st.columns(2)

            with colx:
                st.write(f"**Model: {m1}**")
                out1 = rewrite_resume(
                    resume_text, jd_text,
                    TEMPLATES[template_name], BASE_SYSTEM, m1, temperature,
                )
                st.text_area("", out1, height=320)

            with coly:
                st.write(f"**Model: {m2}**")
                out2 = rewrite_resume(
                    resume_text, jd_text,
                    TEMPLATES[template_name], BASE_SYSTEM, m2, temperature,
                )
                st.text_area("", out2, height=320)

# ===== Footer =====
st.markdown("---")
st.caption("Built by Max Lee — Powered by OpenAI GPT & SentenceTransformers")
# === Evaluation inside the app (reads eval_pairs.csv from repo) ===
with st.expander("📊 Evaluation (rank accuracy)"):
    st.write("Runs metrics using eval_pairs.csv + files under data/resumes and data/jds.")
    if st.button("Run evaluation now"):
        import subprocess
        try:
            out = subprocess.run(
                ["python", "eval_ranking.py"],
                capture_output=True, text=True, check=True
            )
            st.code(out.stdout or "(no stdout)")
            if out.stderr:
                st.code(out.stderr)
        except Exception as e:
            st.error(f"Couldn't run evaluation: {e}")
            st.info("If this fails on Streamlit Cloud, run locally: `python eval_ranking.py`.")

