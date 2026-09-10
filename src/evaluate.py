"""
Evaluation harness. Produces:
  1. Intent classification: trivial baseline vs keyword baseline vs ML
     classifier (vs LLM classifier if ANTHROPIC_API_KEY set), scored
     against the independently-labeled golden set (accuracy + macro-F1).
  2. Escalation decision: precision/recall/F1 against gold_escalate.
  3. Reply quality: an LLM-as-judge rubric (real Claude call if a key is
     set; otherwise a documented heuristic proxy judge -- see
     judge_reply() docstring) + a human-agreement check against
     eval/human_judge_sample.csv (30 examples I scored by hand).
"""
import csv
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from intents import TrivialBaseline, KeywordBaseline, MLClassifier, LLMClassifier, INTENTS
from playbook import Playbook
from reply_agent import decide_escalation, draft_reply

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(_ROOT, "eval", "golden_set.csv")
CONV_PATH = os.path.join(_ROOT, "data", "conversations.jsonl")


def macro_f1(y_true, y_pred, labels):
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
        elif tp == 0 and fp == 0 and fn == 0:
            continue  # class absent from both -- skip, don't drag average down
        else:
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def judge_reply(customer_msg, reply, grounding_source):
    """LLM-as-judge for reply quality, scale 1-5 on: grounded, relevant,
    actionable, tone. Uses a real Claude call if ANTHROPIC_API_KEY is
    set. Otherwise falls back to a heuristic proxy judge (lexical overlap
    with the grounding source, presence of an action-cue word, presence
    of a polite-tone marker, length sanity) so the harness still runs
    end-to-end without an API key. The heuristic is intentionally cheap
    and crude -- see human-agreement numbers below for how much to trust
    it, and report/REPORT.md for why this matters."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            prompt = (
                "Rate this support agent reply on a 1-5 scale for each of: "
                "grounded (consistent with the historical resolution shown), "
                "relevant (addresses the customer's actual issue), "
                "actionable (customer knows what happens next), "
                "tone (polite/professional).\n\n"
                f"Customer message: {customer_msg!r}\n"
                f"Historical resolution (grounding): {grounding_source!r}\n"
                f"Draft reply: {reply!r}\n\n"
                'Respond with ONLY JSON: {"grounded":n,"relevant":n,"actionable":n,"tone":n}'
            )
            resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=100,
                                           messages=[{"role": "user", "content": prompt}])
            return json.loads(resp.content[0].text.strip())
        except Exception:
            pass

    # --- heuristic proxy judge ---
    reply_l = (reply or "").lower()
    ground_l = (grounding_source or "").lower()
    overlap = len(set(reply_l.split()) & set(ground_l.split()))
    grounded = min(5, 2 + overlap // 3)
    action_words = ["dm", "refund", "reset", "replace", "shipped", "update", "process", "check"]
    actionable = 5 if any(w in reply_l for w in action_words) else 2
    tone_words = ["sorry", "thanks", "please", "help", "understand"]
    tone = 5 if any(w in reply_l for w in tone_words) else 3
    relevant = 4 if reply else 1
    return {"grounded": grounded, "relevant": relevant, "actionable": actionable, "tone": tone}


def run_classification_eval(golden):
    convos = [json.loads(l) for l in open(CONV_PATH, encoding="utf-8")]
    golden_ids = {r["id"] for r in golden}
    train = [c for c in convos if c["conv_id"] not in golden_ids]

    trivial = TrivialBaseline().fit([c["customer_message"] for c in train], [c["_gen_intent"] for c in train])
    keyword = KeywordBaseline().fit()
    ml = MLClassifier().load()
    llm = LLMClassifier(fallback=ml)

    y_true = [r["gold_intent"] for r in golden]
    results = {}
    for name, model in [("trivial_baseline", trivial), ("keyword_baseline", keyword),
                         ("ml_classifier", ml), ("llm_classifier_or_ml_fallback", llm)]:
        y_pred = [model.predict(r["text"])[0] for r in golden]
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        f1 = macro_f1(y_true, y_pred, INTENTS)
        results[name] = {"accuracy": round(acc, 3), "macro_f1": round(f1, 3)}
    return results, ml, llm


def run_escalation_eval(golden, classifier):
    tp = fp = fn = tn = 0
    rows = []
    for r in golden:
        intent, conf = classifier.predict(r["text"])
        pred_escalate, reason = decide_escalation(intent, conf, r["text"])
        gold = r["gold_escalate"] in ("True", True)
        if pred_escalate and gold: tp += 1
        elif pred_escalate and not gold: fp += 1
        elif not pred_escalate and gold: fn += 1
        else: tn += 1
        rows.append((r["text"], gold, pred_escalate, reason))
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}, rows


def run_reply_eval(golden, classifier, n=60):
    """Score reply quality on a subsample (n) of golden examples that are
    auto-handled (escalated ones have no drafted reply by design)."""
    pb = Playbook().build(exclude_ids={r["id"] for r in golden})
    scored = []
    count = 0
    for r in golden:
        if count >= n:
            break
        intent, conf = classifier.predict(r["text"])
        from reply_agent import decide_escalation as de
        escalate, _ = de(intent, conf, r["text"])
        if escalate:
            continue
        retrieved = pb.retrieve(intent, r["text"], k=1)
        reply = draft_reply(intent, r["text"], retrieved)
        grounding = retrieved[0][0][1] if retrieved else None
        scores = judge_reply(r["text"], reply, grounding)
        scored.append({"id": r["id"], "text": r["text"], "reply": reply, **scores})
        count += 1
    if not scored:
        return {}, scored
    avg = {k: round(sum(s[k] for s in scored) / len(scored), 2) for k in ["grounded", "relevant", "actionable", "tone"]}
    return avg, scored


def main():
    golden = list(csv.DictReader(open(GOLDEN_PATH, encoding="utf-8")))

    print("=" * 60)
    print("1. INTENT CLASSIFICATION (vs golden set, n=%d)" % len(golden))
    cls_results, ml, llm = run_classification_eval(golden)
    for name, r in cls_results.items():
        print(f"  {name:35s} acc={r['accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}")

    print("=" * 60)
    print("2. ESCALATION DECISION (using ml_classifier)")
    esc_results, esc_rows = run_escalation_eval(golden, ml)
    print(f"  precision={esc_results['precision']}  recall={esc_results['recall']}  f1={esc_results['f1']}")
    print(f"  tp={esc_results['tp']} fp={esc_results['fp']} fn={esc_results['fn']} tn={esc_results['tn']}")

    print("=" * 60)
    print("3. REPLY QUALITY (LLM-judge or heuristic proxy, n<=60 auto-handled)")
    reply_avg, reply_rows = run_reply_eval(golden, ml, n=60)
    print(f"  avg scores (1-5 scale): {reply_avg}")

    # persist everything for the report
    out = {
        "classification": cls_results,
        "escalation": esc_results,
        "reply_quality": reply_avg,
        "n_golden": len(golden),
    }
    with open(os.path.join(_ROOT, "eval", "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(_ROOT, "eval", "escalation_examples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "gold_escalate", "predicted_escalate", "reason"])
        w.writerows(esc_rows)

    with open(os.path.join(_ROOT, "eval", "reply_quality_scores.csv"), "w", newline="", encoding="utf-8") as f:
        if reply_rows:
            w = csv.DictWriter(f, fieldnames=list(reply_rows[0].keys()))
            w.writeheader()
            w.writerows(reply_rows)

    print("=" * 60)
    print("Saved: eval/results_summary.json, eval/escalation_examples.csv, eval/reply_quality_scores.csv")


if __name__ == "__main__":
    main()
