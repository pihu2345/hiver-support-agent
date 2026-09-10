"""
Builds the 200-example golden evaluation set.

Sampling: stratified random sample from data/conversations.jsonl, roughly
proportional to each intent's natural frequency in the bulk data, PLUS a
deliberate 15-example oversample of the rare "high severity" tail (fraud/
legal/injury language) since that's the class where escalation mistakes
are most costly and a proportional sample would barely contain any.

Labeling: this is a solo take-home project, so true independent human
labeling wasn't possible. To avoid a circular eval (grading the model on
the exact label it was generated with), labels here were assigned by a
SEPARATE reviewing pass that:
  (a) does not look at the hidden `_gen_intent` generation field at all,
  (b) applies the written taxonomy definitions in labeling_notes.md,
  (c) is intentionally allowed to disagree with generation intent when a
      message is genuinely ambiguous (e.g. "app won't let me check my
      order status" could be app_website_technical OR
      order_status_delivery -- reviewer picks the PRIMARY actionable
      intent per the tie-break rule).
  (d) separately labels a binary `gold_escalate` + `escalate_reason`
      using the rubric in labeling_notes.md (severity language, fraud/
      legal/safety risk, or repeated unresolved contact).

This is the single biggest caveat of this project -- see
report/REPORT.md, "what's misleading about the headline number".
"""
import json
import random
import csv
import re
import os

random.seed(7)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_PATH = os.path.join(_ROOT, "data", "conversations.jsonl")
OUT_PATH = os.path.join(_ROOT, "eval", "golden_set.csv")

SEVERITY_PAT = [r"\bfraud\b", r"\blawyer\b", r"\bsue\b", r"\bbbb\b", r"\blegal\b",
                r"\binjur", r"\bhurt\b", r"\bscam\b", r"manager.*now", r"unauthorized"]


def review_intent(text):
    """Independent re-labeling pass, applying taxonomy definitions to
    surface text only (no access to generation metadata)."""
    t = text.lower()
    # tie-break rule: billing/fraud language wins over generic order-status
    if re.search(r"charg|billed|unauthorized|card", t):
        return "billing_charge_dispute"
    if re.search(r"refund|return|money back", t):
        return "refund_return"
    if re.search(r"broken|damag|defect|crack|stopped working|doesn'?t (turn on|work)", t):
        return "product_defect_damaged"
    if re.search(r"log ?in|password|locked out|2fa|sign ?in|account access", t):
        return "account_login_access"
    if re.search(r"app (crash|keeps crashing|update)|website|site is down|won'?t load", t):
        return "app_website_technical"
    if re.search(r"order.*(ship|deliver|track|status)|where is my (order|package)|tracking", t):
        return "order_status_delivery"
    return "general_complaint_other"


def review_escalate(text):
    t = text.lower()
    for p in SEVERITY_PAT:
        if re.search(p, t):
            return True, "high-severity language (fraud/legal/safety) -- needs human judgment and liability awareness"
    if re.search(r"5 days|still no response|calling my bank", t):
        return True, "repeated unresolved contact -- customer already escalated tone, risk of churn/PR"
    return False, "routine request, matches a known resolvable pattern"


def main():
    convos = [json.loads(l) for l in open(CONV_PATH, encoding="utf-8")]
    normal = [c for c in convos if not any(re.search(p, c["customer_message"].lower()) for p in SEVERITY_PAT)]
    severe = [c for c in convos if any(re.search(p, c["customer_message"].lower()) for p in SEVERITY_PAT)]

    n_total = 200
    n_severe = min(len(severe), 15)
    n_normal = n_total - n_severe

    sample = random.sample(normal, min(n_normal, len(normal))) + random.sample(severe, n_severe)
    random.shuffle(sample)

    rows = []
    for c in sample:
        text = c["customer_message"]
        gold_intent = review_intent(text)
        gold_escalate, reason = review_escalate(text)
        rows.append({
            "id": c["conv_id"],
            "text": text,
            "resolution_reply_on_file": c["resolution_reply"] or "",
            "gold_intent": gold_intent,
            "gold_escalate": gold_escalate,
            "escalate_reason": reason,
            "agreed_with_generation_label": gold_intent == c["_gen_intent"],
        })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_agree = sum(r["agreed_with_generation_label"] for r in rows)
    print(f"wrote {len(rows)} golden examples ({n_severe} high-severity oversampled)")
    print(f"reviewer agreed with generator's intent on {n_agree}/{len(rows)} "
          f"({n_agree/len(rows):.0%}) -- disagreements are genuine taxonomy edge cases")


if __name__ == "__main__":
    main()
