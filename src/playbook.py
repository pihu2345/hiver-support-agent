"""
Builds a retrieval index over historical (customer_message -> resolution_reply)
pairs so replies are GROUNDED in how the brand has actually resolved similar
issues before, rather than hallucinated from scratch.

Approach: TF-IDF nearest-neighbor retrieval, indexed separately per intent
(so we never retrieve a billing resolution for a delivery question even if
it's textually similar). This is intentionally simple/inspectable rather
than an embedding+vector-DB setup -- see decision_log.md for why.
"""
import json
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_PATH = os.path.join(_ROOT, "data", "conversations.jsonl")


class Playbook:
    def __init__(self):
        self.by_intent = defaultdict(list)  # intent -> list of (customer_msg, resolution_reply)
        self.vectorizers = {}
        self.matrices = {}

    def build(self, exclude_ids=None, min_turns=4):
        """min_turns=4 means: opener -> ack -> customer follow-up -> REAL
        resolution. Threads with n_turns<4 only ever got a generic "please
        DM us" ack, not an actual resolution -- including those in the
        grounding pool was a real bug found during eval (see
        report/REPORT.md failure mode #1): retrieval kept surfacing the
        generic ack instead of a substantive resolution, because acks
        vastly outnumber real resolutions and are textually similar
        across every intent."""
        exclude_ids = exclude_ids or set()
        convos = [json.loads(l) for l in open(CONV_PATH, encoding="utf-8")]
        for c in convos:
            if c["conv_id"] in exclude_ids:
                continue  # never let golden-eval examples leak into the playbook
            if not c["resolution_reply"]:
                continue
            if c.get("n_turns", 0) < min_turns:
                continue  # skip ack-only threads, keep only real resolutions
            self.by_intent[c["_gen_intent"]].append((c["customer_message"], c["resolution_reply"]))

        for intent, pairs in self.by_intent.items():
            texts = [p[0] for p in pairs]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            mat = vec.fit_transform(texts)
            self.vectorizers[intent] = vec
            self.matrices[intent] = mat
        return self

    def retrieve(self, intent, text, k=1):
        """Return the k most similar historical (customer_msg, resolution_reply)
        pairs for the given predicted intent. Falls back across all intents
        if this intent has no history (cold start)."""
        pool_intent = intent if intent in self.by_intent and self.by_intent[intent] else None
        if pool_intent is None:
            # cold start fallback: search everything
            all_pairs = [p for pairs in self.by_intent.values() for p in pairs]
            if not all_pairs:
                return []
            texts = [p[0] for p in all_pairs]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(texts)
            mat = vec.transform(texts)
            qv = vec.transform([text])
            sims = cosine_similarity(qv, mat)[0]
            top = sims.argsort()[::-1][:k]
            return [(all_pairs[i], float(sims[i])) for i in top]

        vec = self.vectorizers[intent]
        mat = self.matrices[intent]
        qv = vec.transform([text])
        sims = cosine_similarity(qv, mat)[0]
        top = sims.argsort()[::-1][:k]
        pairs = self.by_intent[intent]
        return [(pairs[i], float(sims[i])) for i in top]
