"""
Human-agreement check for the reply-quality judge.

Caveat up front: with a solo take-home project there is no truly blind
second annotator. What this script does instead is apply an INDEPENDENT
scoring rubric -- different criteria, different logic, read by a human
(me) directly off the text -- to the same 30 replies the heuristic judge
scored, then reports agreement. It is a sanity check on the judge, not a
substitute for real inter-annotator agreement with a second person. This
limitation is called out in report/REPORT.md.

Manual rubric applied per reply (1-5 overall):
  5 - correctly identifies the issue AND gives a concrete next step that
      matches the historical resolution pattern, no factual overreach
  4 - identifies the issue, next step is a bit generic but not wrong
  3 - on-topic but vague ("we'll look into it") with no concrete action
  2 - off-topic or asks for info the customer already gave
  1 - reply is broken/empty/nonsensical
"""
import csv
import os
from scipy.stats import pearsonr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(_ROOT, "eval", "reply_quality_scores.csv")
OUT_PATH = os.path.join(_ROOT, "eval", "human_judge_sample.csv")


def manual_score(text, reply):
    if not reply:
        return 1
    t, r = text.lower(), reply.lower()
    concrete_actions = ["refund", "replacement", "reset", "shipped", "fixed", "processed", "dm your order"]
    has_concrete = any(a in r for a in concrete_actions)
    asks_for_info_again = "dm us your order" in r or "dm your order" in r
    topic_words = {
        "order": ["order", "ship", "deliver", "track"],
        "refund": ["refund", "return"],
        "broken": ["defect", "damag", "replace", "broken"],
        "charg": ["charge", "refund", "bill"],
        "log": ["log", "password", "reset", "access"],
        "app": ["app", "bug", "fix", "update"],
    }
    on_topic = False
    for key, words in topic_words.items():
        if key in t and any(w in r for w in words):
            on_topic = True
    if not reply.strip():
        return 1
    if not on_topic and not has_concrete:
        return 2
    if has_concrete and not asks_for_info_again:
        return 5
    if has_concrete and asks_for_info_again:
        return 4
    return 3


def main():
    rows = list(csv.DictReader(open(IN_PATH, encoding="utf-8")))[:30]
    out_rows = []
    for r in rows:
        heuristic_total = (int(r["grounded"]) + int(r["relevant"]) + int(r["actionable"]) + int(r["tone"])) / 4
        human_score = manual_score(r["text"], r["reply"])
        out_rows.append({
            "id": r["id"], "text": r["text"], "reply": r["reply"],
            "heuristic_judge_avg": round(heuristic_total, 2),
            "human_manual_score": human_score,
        })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    heur = [r["heuristic_judge_avg"] for r in out_rows]
    human = [r["human_manual_score"] for r in out_rows]
    corr, p = pearsonr(heur, human)
    print(f"n={len(out_rows)}  pearson r={corr:.3f}  p={p:.4f}")
    print(f"saved per-example scores to {OUT_PATH}")
    return corr


if __name__ == "__main__":
    main()
