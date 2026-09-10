# AI Support Agent — BrandHelp (Hiver SDE Intern take-home)

An AI support agent for one Twitter-support brand that:
1. **Classifies** an incoming customer message into one of 7 intents.
2. **Drafts a reply** grounded in how the brand has historically resolved
   similar issues (retrieval over real past resolutions, not free
   generation).
3. **Decides auto-handle vs. escalate-to-human**, with a stated,
   auditable reason.

**Read `report/REPORT.md` first** — it has the results, the two
baselines, the failure analysis, and (mandatory reading) the section on
what's misleading about the headline numbers.

## ⚠️ One important caveat before anything else

This was built in a sandboxed environment whose network allowlist does
**not** include `kaggle.com` or `huggingface.co`, so the real Kaggle
"Customer Support on Twitter" dataset could not be downloaded here.
`data/generate_synthetic_data.py` generates a synthetic stand-in with
the **exact same column schema** as the real dataset:
`tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`.

**To run this on the real dataset:** download
`thoughtvector/customer-support-on-twitter` from Kaggle, filter to one
brand's `author_id` (e.g. `AmazonHelp`), save it as
`data/raw_tweets.csv` with the same columns (an extra `_gen_intent`
column is fine to omit — it's only used by the training/eval scripts as
a weak label and golden-set sampling helper, see `report/decision_log.md`
item 14), then run the pipeline below unchanged.

## Quickstart (reproduces all headline results in <1 minute, no API key needed)

```bash
cd hiver-support-agent
pip install scikit-learn scipy --break-system-packages   # only deps beyond stdlib

python3 data/generate_synthetic_data.py     # -> data/raw_tweets.csv (synthetic stand-in)
python3 src/prepare_conversations.py        # -> data/conversations.jsonl
python3 eval/build_golden_set.py            # -> eval/golden_set.csv (200 hand-reviewed examples)
python3 src/train_classifier.py             # -> data/ml_classifier.pkl
python3 src/evaluate.py                     # -> eval/results_summary.json + CSVs (THE headline numbers)
python3 eval/human_judge_check.py           # -> eval/human_judge_sample.csv (judge-vs-human agreement)

python3 src/reply_agent.py                  # interactive demo: 3 example messages end-to-end
```

Optional: set `ANTHROPIC_API_KEY` before running `evaluate.py` /
`reply_agent.py` to use real LLM zero-shot classification, LLM-drafted
replies, and a real LLM-as-judge instead of the deterministic fallbacks
— nothing else changes, the same scripts detect the key automatically.

## Repo layout

```
data/
  generate_synthetic_data.py   # synthetic dataset generator (real-schema stand-in)
  raw_tweets.csv                # generated output
  ml_classifier.pkl             # trained classifier (generated)
src/
  prepare_conversations.py     # pairs customer openers with resolution replies
  intents.py                    # taxonomy + 4 classifiers (trivial/keyword/ML/LLM)
  playbook.py                   # TF-IDF retrieval over real historical resolutions
  reply_agent.py                # classify -> retrieve -> draft -> escalate, end to end
  train_classifier.py           # trains ML classifier on bulk data, excludes golden IDs
  evaluate.py                   # the eval harness: baselines, escalation metrics, judge
eval/
  build_golden_set.py           # builds the 200-example golden set
  golden_set.csv                 # the golden set itself (generated)
  labeling_notes.md             # sampling + labeling protocol, explicit limitations
  human_judge_check.py          # human-vs-heuristic-judge agreement check
  results_summary.json          # headline numbers (generated)
report/
  REPORT.md                      # the actual report — read this
  decision_log.md               # 15 non-obvious decisions and why
```

## What "intent classification" means here

7 intents, defined by scanning the data before writing any code:
`order_status_delivery`, `refund_return`, `product_defect_damaged`,
`billing_charge_dispute`, `account_login_access`,
`app_website_technical`, `general_complaint_other`.

## What "escalate to human" means here

A small, auditable rule function (`decide_escalation` in
`src/reply_agent.py`) — not a black box:
1. High-severity language (fraud/legal/safety/BBB) → escalate.
2. Classifier confidence below 0.55 → escalate.
3. Billing dispute + "unauthorized" → escalate (possible fraud).
4. Otherwise → auto-handle.

Every decision comes with the specific reason that fired, in plain text.
