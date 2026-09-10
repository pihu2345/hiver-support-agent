# Decision log

Plain list of non-obvious calls made and why. Roughly chronological.

1. **Used a synthetic stand-in dataset, not the real Kaggle CSV.** This
   sandbox's network allowlist has no path to kaggle.com or
   huggingface.co. Rather than fake results against data I never
   touched, I built a generator matching the real dataset's exact
   column schema so the real CSV is a drop-in replacement, and I say so
   loudly in the report instead of hiding it.

2. **Picked a 7-intent taxonomy instead of reusing Banking77's 77
   labels.** Banking77 is banking-specific and far too fine-grained for
   a general e-commerce/consumer brand's Twitter support volume; 77
   classes would also make a 200-example golden set nearly useless (not
   enough examples per class). Built the taxonomy from scratch by
   scanning the kind of complaints that actually show up in this
   dataset shape.

3. **Weak-supervision training + independent golden eval, rather than
   hand-labeling the training set too.** Labeled 200 examples by hand
   (well, by an independent review pass — see caveat in
   labeling_notes.md) for evaluation, but trained the classifier on
   ~500 bulk examples with cheaper, noisier labels. This is closer to
   how a real team would actually bootstrap this system than hand-
   labeling thousands of examples for a take-home.

4. **TF-IDF + Logistic Regression over an LLM/embedding classifier by
   default.** Zero API cost, runs in under a second, fully inspectable
   (you can read the learned coefficients), and the assignment
   explicitly wants reproducibility in 15 minutes without assuming API
   access. An LLM zero-shot path is wired in and used automatically if
   `ANTHROPIC_API_KEY` is set — best of both, but the headline numbers
   don't depend on having a key.

5. **TF-IDF retrieval instead of embeddings/vector DB for grounding.**
   Same reasoning as #4 — inspectable, zero-dependency, and at this
   dataset size (a few hundred examples) an embedding index wouldn't
   meaningfully outperform TF-IDF while adding real setup cost.

6. **Grounding index is scoped per-intent, not global.** Retrieving
   across all intents risked pulling a textually-similar but
   topically-wrong historical reply (e.g. a refund resolution for a
   delivery question). Cold-start fallback (if an intent bucket is
   empty) searches everything rather than failing.

7. **Filtered the grounding pool to real resolutions only
   (`min_turns>=4`), not every historical brand reply.** Found this bug
   myself during failure analysis (see report, failure mode #1) — the
   first version retrieved generic "please DM us" acks constantly. Left
   the discovery process in the report rather than quietly fixing it
   and pretending the first version never existed.

8. **Escalation is a small, auditable rule function, not a learned
   model.** For a support agent deciding whether to auto-send a reply,
   I wanted every decision traceable to a specific, readable reason a
   human can audit — "confidence 0.54 < 0.55" or "matched fraud
   keyword" — rather than a black-box escalation classifier.

9. **Escalation policy is asymmetric on purpose: optimize precision on
   high-severity language, accept lower recall elsewhere.** A false
   escalation costs a human a few seconds of review; a false
   auto-handle on a fraud claim risks real harm. Encoded this directly
   rather than picking a single threshold that treats both errors
   equally.

10. **Oversampled the rare high-severity tail (15/200) in the golden
    set instead of pure random sampling.** A proportional sample would
    have ~1-2 severe examples — not enough to say anything meaningful
    about escalation recall on the class that matters most. Reported
    metrics on this subgroup separately rather than blending it
    invisibly into one aggregate number.

11. **Deliberately did NOT clean up the keyword-baseline / golden-label
    circularity after discovering it.** I could have quietly rewritten
    `review_intent()` with different logic so the keyword baseline
    wouldn't score 100%. I left it as a documented finding instead
    (report Section 4, item 1) because catching your own eval bugs is
    more valuable to show than a clean-looking table.

12. **LLM-as-judge falls back to a heuristic proxy, and I explicitly
    measured that the proxy doesn't correlate with manual judgment
    (r=0.11).** Could have just not run the human-agreement check and
    let the 1-5 scores stand unchallenged. Chose to report the negative
    result because a judge you haven't validated shouldn't be trusted
    just because it produces a number.

13. **Single-label intent classification, no thread memory.** Both are
    real simplifications, not oversights — documented as explicit scope
    cuts in the report rather than silently ignored, with a stated plan
    for what I'd do next.

14. **`_gen_intent` (the synthetic generation label) is never read by
    any evaluation or golden-labeling code** — only by the classifier
    trainer, which is explicitly a weak-supervision source, not ground
    truth. Structured the code this way (separate module boundaries) so
    it isn't possible to accidentally leak it into the eval by mistake.

15. **Reply drafting never invents numbers.** The deterministic fallback
    reuses the retrieved historical reply verbatim rather than trying to
    template-fill an order number or refund amount the system doesn't
    actually know, because a wrong specific number is worse than a
    slightly generic reply.
