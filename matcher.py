from sentence_transformers import SentenceTransformer, util
from typing import Dict

_model = None
def _get_model():
    global _model
    if _model is None:
        # small, fast model; auto-downloads on first run
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _model

def compute_match_score(resume_text: str, job_desc: str) -> float:
    """Return 0–100 cosine similarity percentage."""
    m = _get_model()
    emb_resume = m.encode(resume_text, convert_to_tensor=True, normalize_embeddings=True)
    emb_job = m.encode(job_desc, convert_to_tensor=True, normalize_embeddings=True)
    score = float(util.cos_sim(emb_resume, emb_job).item())
    return round(max(0.0, min(1.0, score)) * 100, 2)

def gap_analysis(resume_text: str, job_desc: str) -> Dict[str, list]:
    """Simple keyword gap heuristic to complement embeddings."""
    import re, collections
    def tokens(t):
        toks = re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{1,}", t.lower())
        stop = set('the a an and or of to for with on in from by as at is are be been was were you your our their they them it its this that'.split())
        return [x for x in toks if x not in stop and len(x) >= 2]
    rset = set(tokens(resume_text))
    jtok = tokens(job_desc)
    freq = collections.Counter(jtok)
    missing = [w for w,_ in freq.most_common(50) if w not in rset][:15]
    present = [w for w,_ in freq.most_common(50) if w in rset][:15]
    return {"missing_keywords": missing, "present_keywords": present}
