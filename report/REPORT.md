# AI Support Agent for "BrandHelp" — Report

Brand: a stand-in Twitter support handle called **BrandHelp**, modeled on
the shape of real handles in the Kaggle Customer-Support-on-Twitter
dataset (@AmazonHelp / @AppleSupport style: short informal tweets, an
agent asking to move to DM, a resolution a few turns later).

**Important upfront caveat (also see "misleading number" section below):**
this sandbox's network allowlist does not include kaggle.com or
huggingface.co, so I could not download the real dataset here. Every
script in this repo is written against the *exact* column schema of the
real dataset (`tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id`), so dropping the real CSV
into `data/raw_tweets.csv` reproduces everything unchanged — see README
for the one-line swap. What's in this report is the pipeline validated
on a synthetic, template-generated stand-in. I treat this as the single
biggest limitation of the whole project and don't pretend otherwise.

---

## 1. Problem framing

**What "good" means for this brand:**
- **Classification** should route a message to the right internal
  playbook, not just guess plausibly. A wrong intent means a wrong
  grounding source means a wrong reply — errors compound downstream, so
  I optimized macro-F1 (treats rare intents like account lockouts as
  seriously as common ones like delivery delays) over raw accuracy.
- **Reply drafting** should never say something the brand can't back up
  (a refund amount, a promise, a timeline) unless it's grounded in an
  actual historical resolution. I'd rather send a shorter, more generic
  "please DM your order #" than a fluent but unsupported promise.
- **Escalation** should have near-perfect *precision* on the highest-risk
  category (fraud/legal/safety language) even at some cost to recall
  elsewhere — a human seeing an unnecessary escalation costs a few
  seconds; an auto-reply to a fraud claim costs trust and possibly
  liability.

**What I chose not to build:**
- No real-time Twitter API integration — out of scope for a take-home,
  and it would add zero signal about whether the classification/
  grounding/escalation logic is sound.
- No fine-tuned LLM — a TF-IDF+LogisticRegression classifier and
  TF-IDF retrieval-based grounding get most of the value here at near-
  zero cost and are fully inspectable; an LLM zero-shot classifier and
  LLM-rewritten replies are wired in as an *optional* upgrade path
  (`ANTHROPIC_API_KEY` env var) but the headline numbers below don't
  depend on having one, by design — the assignment wants this
  reproducible in 15 minutes without a paid key.
- No multi-intent / multi-label classification, even though real
  tweets clearly have compound issues ("my order is late AND I was
  double-charged"). Single-label was a scope cut, not a belief that
  it's the right long-term design — see "next week."
- No conversation-level memory across a multi-turn thread (each message
  is scored independently). Real support agents use thread history
  heavily; this is the single most consequential thing skipped for time.

---

## 2. Results vs. baselines

Evaluated on the 200-example golden set (`eval/golden_set.csv`),
independently labeled — see `eval/labeling_notes.md` for the protocol
and its limits.

### Intent classification

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Trivial baseline (always predict majority class) | 0.120 | 0.031 |
| **Keyword-rule baseline** (hand-written regex) | 1.000 | 1.000 |
| **ML classifier** (TF-IDF + Logistic Regression) | 0.840 | 0.865 |
| LLM zero-shot (falls back to ML classifier without an API key) | 0.840 | 0.865 |

The keyword baseline's perfect score is **not real signal** — see
Section 4, it's the headline example of a misleading number in this
project. Ignore it; the ML classifier's 84.0% / 0.865 macro-F1 is the
trustworthy comparison, and it beats the trivial baseline by a wide
margin (12% accuracy — reflecting how skewed the intent mix is).

### Escalation decision (rule-based, using ML classifier's output)

| Metric | Value |
|---|---|
| Precision | 1.000 (0 false escalations out of 15 predicted) |
| Recall | 0.789 (15/19 gold-escalate cases caught) |
| F1 | 0.882 |

Zero false positives was a deliberate design choice (see Section 1) —
the policy only fires on unambiguous severity language or low
classifier confidence, so it never escalates something routine. The
4 misses are analyzed in Section 3.

### Reply quality (judge scores, 1-5 scale, n=60 auto-handled replies)

| Dimension | Score |
|---|---|
| Grounded (matches historical resolution) | 5.0 |
| Relevant | 4.0 |
| Actionable | 4.8 |
| Tone | 4.8 |

**These numbers should not be trusted at face value** — no
`ANTHROPIC_API_KEY` was available in this sandbox, so this ran on the
heuristic proxy judge, not a real LLM judge. See Section 4: when I
checked the proxy judge against my own manual scoring on 30 examples,
agreement was essentially zero (Pearson r = 0.11, p = 0.55, not
significant). **Do not cite the 1-5 reply-quality numbers as evidence
of anything** until run with a real LLM judge — the harness supports
this out of the box (just set the env var), I simply couldn't verify it
in this environment.

---

## 3. Failure analysis: top 5 failure modes

**1. Grounding pool polluted by generic "ack" messages (found during
this project, since fixed).** Early version of the retrieval index
included every historical brand reply, including the generic "please DM
us your order #" acknowledgment that appears in *every* thread. Because
acks are far more numerous than real resolutions and textually similar
across all intents, TF-IDF retrieval kept surfacing the ack instead of
a substantive resolution — drafts looked fine on paper but a human
reviewer immediately noticed the reply just repeated the customer's own
ask back at them. Fixed by filtering the grounding pool to only
threads with a real follow-up + resolution (`min_turns=4` in
`playbook.py`). Root cause: no distinction in the raw data between
"we acknowledged this" and "we resolved this" — worth flagging for any
real deployment, since the real Kaggle dataset has the identical
ambiguity.

**2. Rare-tail intents get misrouted by the ML classifier because
training data under-represents them.** All 4 "fraud, calling my
lawyer" examples that mention an order number were classified as
`order_status_delivery` (confidence ~0.33-0.62, notably lower than the
0.94+ confidence on clean matches) instead of `billing_charge_dispute`.
These templates are only ~8% of the bulk training data and share
surface vocabulary ("order #NNNNNN") with the much larger delivery
class, so the linear classifier leans on that shared token. This is
exactly the kind of message where a wrong intent is most costly (it's
also usually high-severity), which is why the escalation policy has an
independent, keyword-based severity check that does *not* depend on the
intent classifier at all — the two failure modes partially cancel out
in practice (all 4 of these were still correctly escalated, just for
the "severity language" reason rather than "billing" reason).

**3. Escalation recall misses "slow-burn" frustration without a hard
trigger word.** The 4 missed gold-escalate cases were routine-sounding
delivery complaints ("still hasn't shipped, been 5 days??") that the
labeling rubric flagged as escalate-worthy purely because they matched
a loose "N days" regex (see `eval/labeling_notes.md`) meant to catch
repeated-unresolved-contact language. On inspection, these are
genuinely borderline — a first-time "5 days late" complaint is arguably
fine to auto-handle with a tracking check. This is as much a labeling-
rubric weakness as a system failure; I'm listing it here rather than
quietly tightening the label after seeing the eval, because that's the
honest way to report it.

**4. Taxonomy-definition drift between weak training labels and the
independent golden review.** "Any update on order #X, it's stuck in
transit forever" was generated under `order_status_delivery` but the
golden-review taxonomy definition (delivery = contains "ship/deliver/
track/status") doesn't technically match "stuck in transit," so the
independent reviewer labeled it `general_complaint_other`. The ML
classifier (trained on the generation label) confidently predicts
`order_status_delivery` — which is arguably *more* correct in spirit,
but counts as an "error" against the golden set. This is a reminder
that a hand-written taxonomy's literal keyword definitions can diverge
from its intended meaning, and it inflates the apparent error count.

**5. The LLM-judge proxy doesn't actually measure reply quality.**
Documented in Section 4 — the heuristic judge rewarded lexical overlap
with the retrieved historical reply and the presence of action-cue
words, which correlated poorly (r=0.11) with my own manual judgment of
whether the reply actually helped a specific customer. Concretely, the
proxy gave a 4.75/5 average to replies that were the same generic
"please DM your order #" template regardless of whether the customer
had *already* said they'd DMed twice with no response — a real judge
(or human) would penalize that as tone-deaf and unhelpful.

---

## 4. What is misleading about my headline number

Several things, and I'd rather list them than let a clean-looking table
hide them:

1. **The keyword baseline's 100% accuracy is a labeling artifact, not a
   real result.** The golden-set intent labels were generated by a
   review function (`review_intent()`) using regex rules that are
   *identical in structure* to the keyword-baseline classifier
   (`KeywordBaseline` in `src/intents.py`) — both are "if billing words,
   then billing; elif refund words, then refund; ..." Since the same
   logic both produced the ground truth and produced the prediction,
   the keyword baseline is definitionally correct on this eval set. It
   would almost certainly score much worse on real tweets it wasn't
   designed around. **The real story is that the ML classifier (84%
   accuracy) is a *harder* comparison than "beats a keyword list" makes
   it sound**, because the eval set has a structural bias in the
   keyword baseline's favor that the ML model doesn't share.
2. **The entire dataset is synthetic**, generated from ~30 hand-written
   templates. Real tweets have typos, sarcasm, multiple issues per
   message, non-English text, memes, and messages that reference
   earlier tweets by other users in a thread — none of that exists here.
   Every number in this report is an upper bound on what would happen
   on the real Kaggle data, not an estimate of it.
3. **The "reply quality" scores are from a heuristic proxy judge that I
   showed (Section 3, failure #5) doesn't correlate with my own manual
   judgment.** They shouldn't be quoted as if a real LLM-as-judge or a
   human evaluated reply quality — that check simply wasn't possible to
   run for real in this sandbox (no API key, no third annotator), and
   I'd rather say that explicitly than present r=0.11 agreement as if
   it were r=0.7.
4. **Escalation precision of 1.0 looks stronger than it is** — it's easy
   to get zero false positives when the escalation rule barely covers
   any of the message space (only fires on explicit severity keywords
   or a confidence floor). A rule that escalated *everything* would also
   have "perfect precision" on a differently-shaped confusion matrix
   — precision alone is not a complete story here, recall (0.79) and the
   4 concrete misses in Section 3 matter more.
5. **The 200-example golden set is not independent in the strict
   sense** — same author who built the taxonomy and the templates also
   wrote the review labels, same day, with the templates fresh in mind.
   85% agreement between the "independent" review and the generation
   label is a weaker signal of taxonomy quality than it would be if a
   second person had done the labeling blind.

---

## 5. What I'd do with one more week

1. **Get the real dataset.** Download the actual Kaggle CSV, filter to
   one real brand handle, and re-run every script unchanged (the schema
   match is intentional) — this alone would invalidate or validate
   everything else above.
2. **Real LLM-as-judge + a second human labeler**, even if it's a
   friend spending an hour, to get an actual inter-annotator agreement
   number instead of a self-consistency check.
3. **Multi-label intent classification** — real support tweets often
   have 2 problems in one message; forcing single-label throws away
   information the routing system needs.
4. **Thread-level context for the classifier and drafter** — right now
   every message is scored in isolation; a customer's 3rd message in a
   thread should be read differently from their 1st.
5. **Calibrate the escalation confidence threshold properly** — 0.55 was
   picked by eyeballing a handful of examples, not by sweeping a
   precision/recall curve against the golden set; with more time I'd
   grid-search the threshold and report the full curve, not one point.
6. **Detect and fix the "5 days" over-broad escalation trigger** in the
   golden-set rubric itself (failure mode #3) rather than reporting it
   as a caveat.

---

## Repro summary

```
python3 data/generate_synthetic_data.py     # ~1s  -> data/raw_tweets.csv
python3 src/prepare_conversations.py        # ~1s  -> data/conversations.jsonl
python3 eval/build_golden_set.py            # ~1s  -> eval/golden_set.csv
python3 src/train_classifier.py             # ~1s  -> data/ml_classifier.pkl
python3 src/evaluate.py                     # ~5s  -> eval/results_summary.json + CSVs
python3 eval/human_judge_check.py           # ~1s  -> eval/human_judge_sample.csv
```
Full run: well under a minute on a laptop, no API key required.
