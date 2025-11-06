# eval_ranking.py
import pandas as pd, numpy as np
from pathlib import Path
from matcher import compute_match_score

LABEL_MAP = {0:"Poor", 1:"Medium", 2:"Good"}

def read_text(p: str) -> str:
    return Path(p).read_text(encoding="utf-8", errors="ignore")

def precision_at_k(labels_sorted, k=3):
    topk = labels_sorted[:k]
    positives = sum(1 for y in topk if y == 2)  # count "Good"
    return positives / k

def mean_reciprocal_rank(relevance_list):
    # relevance_list: 1 for Good else 0, in ranked order
    for i, rel in enumerate(relevance_list, start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0

def main(csv_path="eval_pairs.csv"):
    df = pd.read_csv(csv_path)
    results, p3s, mrrs = [], [], []
    for jd_path, group in df.groupby("jd_path"):
        jd_text = read_text(jd_path)
        scored = []
        for _, row in group.iterrows():
            r_text = read_text(row["resume_path"])
            score = compute_match_score(r_text, jd_text)
            scored.append((row["resume_path"], score, int(row["label"])))
        scored.sort(key=lambda x: x[1], reverse=True)
        y_sorted = [lab for _,_,lab in scored]
        rel_bin = [1 if y==2 else 0 for y in y_sorted]
        p3s.append(precision_at_k(y_sorted, k=3))
        mrrs.append(mean_reciprocal_rank(rel_bin))
        results.extend([(jd_path, rp, round(s,2), LABEL_MAP[lab]) for rp,s,lab in scored])

    out = pd.DataFrame(results, columns=["JD","Resume","PredictedScore","TrueLabel"])
    print("\nPer-JD ranking results:\n", out.to_string(index=False))
    print("\nSummary metrics:")
    print(f"Precision@3 (avg): {np.mean(p3s):.3f}")
    print(f"MRR (avg):         {np.mean(mrrs):.3f}")

if __name__ == "__main__":
    main()
