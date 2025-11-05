import streamlit as st
from utils.text_io import read_any, clean_text
from matcher import compute_match_score, gap_analysis
from rewriter import rewrite_resume

st.set_page_config(page_title="Job Matcher & CV Optimizer", page_icon="🎯", layout="wide")

st.title("🎯 AI-Powered Job Matcher & CV Optimizer")
st.caption(
    "Upload/Paste your resume and a target Job Description. "
    "Get a Job Match Score, keyword gaps, and an LLM-tailored resume. "
    "This app does not store your data."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    r_file = st.file_uploader("Upload resume (.pdf/.docx/.txt) or paste below",
                              type=["pdf","docx","txt"], key="resume_file")
    resume_text = st.text_area("Or paste resume text", height=260, key="resume_text")
    if r_file:
        bytes_data = r_file.read()
        resume_text = read_any(bytes_data, r_file.name)

with col2:
    st.subheader("Job Description")
    j_file = st.file_uploader("Upload JD (.pdf/.docx/.txt) or paste below",
                              type=["pdf","docx","txt"], key="jd_file")
    jd_text = st.text_area("Or paste job description", height=260, key="jd_text")
    if j_file:
        bytes_data = j_file.read()
        jd_text = read_any(bytes_data, j_file.name)

st.markdown("---")
if st.button("Analyze & Optimize", type="primary", use_container_width=True):
    if not resume_text or not jd_text:
        st.error("Please provide both resume and job description.")
    else:
        resume_text = clean_text(resume_text)
        jd_text = clean_text(jd_text)

        with st.spinner("Computing Job Match Score..."):
            score = compute_match_score(resume_text, jd_text)
            gaps = gap_analysis(resume_text, jd_text)
        st.success(f"Job Match Score: **{score}%**")

        with st.expander("Keyword Gap Analysis", expanded=True):
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Missing (consider addressing):**")
                if gaps["missing_keywords"]:
                    st.write(", ".join(gaps["missing_keywords"]))
                else:
                    st.write("None detected — great alignment!")
            with colB:
                st.markdown("**Already Present:**")
                st.write(", ".join(gaps["present_keywords"]))

        with st.spinner("Rewriting resume with LLM (facts preserved)..."):
            optimized = rewrite_resume(resume_text, jd_text)

        st.subheader("Optimized Resume (LLM)")
        st.text_area("", optimized, height=420)
        st.download_button("Download Optimized Resume (.txt)",
                           optimized, file_name="optimized_resume.txt",
                           use_container_width=True)

st.markdown(
    "**Privacy:** Uploaded content is processed in-memory for this session only.  \n"
    "**Disclaimer:** This tool improves presentation and alignment; it does not alter your actual qualifications."
)
