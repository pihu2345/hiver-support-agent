# Golden set: sampling and labeling notes

**File:** `eval/golden_set.csv` (200 examples)
**Builder:** `eval/build_golden_set.py`

## Sampling
- Universe: `data/conversations.jsonl`, one row per customer thread-opener (694 total).
- 185 examples: simple random sample from the "normal" pool (no severity language).
- 15 examples: deliberately oversampled from the rare high-severity tail
  (fraud / legal / BBB / injury language) — a proportional random sample
  would contain ~1-2 of these, which isn't enough to evaluate the
  escalation policy's recall on the class that matters most.
- This is stratified-with-oversampling, not i.i.d. — reported metrics on
  the severe subgroup are therefore reported separately, never blended
  silently into one "escalation recall" number without the n.

## Labeling (intent)
- Applied 7-way taxonomy defined in `src/intents.py`, written from
  scanning ~100 raw threads before building anything else.
- Labels were assigned by a review pass in `review_intent()` that reads
  ONLY the raw text — it does not have access to the hidden generation
  label (`_gen_intent`). Tie-break order when a message could fit two
  buckets: billing/fraud language > refund/return > product
  defect > account/login > app/site technical > order status >
  otherwise general_complaint_other.
- Sanity check: the review labels agreed with the (separately produced)
  generation labels on 170/200 (85%) of examples. The 15% disagreement
  is not noise — it's genuinely ambiguous phrasing (e.g. "any update on
  my order, it's stuck in transit forever" doesn't contain any of the
  explicit tracking/shipping keywords in the taxonomy's own definition,
  so by the letter of the taxonomy it falls back to
  general_complaint_other even though a human agent would obviously
  treat it as a delivery question). This mismatch is itself one of our
  top-5 failure modes — see report/REPORT.md.

## Labeling (escalation)
- Binary `gold_escalate` + free-text `escalate_reason`, from
  `review_escalate()`: high-severity language (fraud/legal/safety/BBB) OR
  repeated-unresolved-contact phrasing ("5 days", "still no response",
  "calling my bank") escalates; everything else doesn't.
- Known weakness, found during failure analysis: the "5 days" / repeated-
  contact trigger is looser than intended and flags some routine
  "still hasn't shipped, 5 days" delivery complaints as escalate-worthy
  even though a competent agent would just check the tracking number.
  We left this in rather than quietly tightening it post-hoc, and instead
  report it as a labeling-rubric weakness in the report.

## What this is NOT
This is a solo take-home project — there was no second, blind human
annotator. The review pass above is independent of the generation
labels in the sense that it doesn't read them, but it was still written
by the same person, on the same day, informed by having seen the
generation templates while building the taxonomy. Treat the 85%
agreement number as an internal consistency check, not inter-annotator
agreement in the formal sense. This caveat is repeated in
`report/REPORT.md`.
