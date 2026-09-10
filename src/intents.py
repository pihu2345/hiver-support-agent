"""
Intent taxonomy + three classifiers of increasing sophistication:

  1. TrivialBaseline   - always predicts the single most common intent.
  2. KeywordBaseline   - simple hand-written keyword/regex rules per intent.
  3. MLClassifier      - TF-IDF + Logistic Regression, trained on BULK
                         weakly-labeled conversations (the `_gen_intent`
                         field acts as a noisy/weak label here -- this
                         mirrors a realistic setup where you bootstrap
                         training labels cheaply and only hand-label a
                         small, independent golden set for evaluation).
  4. LLMClassifier     - optional zero-shot classifier via the Anthropic
                         API (only used if ANTHROPIC_API_KEY is set).
                         Falls back to MLClassifier if no key / call fails.

Only classifiers 3 and 4 are candidates for "the system"; 1 and 2 are the
two required baselines (trivial + simple).
"""
import os
import re
import json
import pickle
from collections import Counter

INTENTS = [
    "order_status_delivery",
    "refund_return",
    "product_defect_damaged",
    "billing_charge_dispute",
    "account_login_access",
    "app_website_technical",
    "general_complaint_other",
]

KEYWORD_RULES = {
    "billing_charge_dispute": [r"charg", r"billed", r"unauthorized", r"double charge", r"card"],
    "refund_return": [r"refund", r"return", r"money back"],
    "product_defect_damaged": [r"broken", r"damag", r"defect", r"crack", r"stopped working", r"doesn'?t (turn on|work)"],
    "order_status_delivery": [r"order.*(ship|deliver|track|status)", r"where is my (order|package)", r"tracking"],
    "account_login_access": [r"log ?in", r"password", r"locked out", r"2fa", r"sign ?in", r"account access"],
    "app_website_technical": [r"app (crash|keeps crashing|update)", r"website", r"site is down", r"won'?t load"],
}

SEVERITY_KEYWORDS = [
    r"\bfraud\b", r"\blawyer\b", r"\bsue\b", r"\bbbb\b", r"\blegal\b",
    r"\binjur", r"\bhurt\b", r"\bscam\b", r"\bmanager\b.*\bnow\b",
    r"unauthorized",
]


class TrivialBaseline:
    """Always predicts the majority intent seen in training data."""
    def fit(self, texts, labels):
        self.majority = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, text):
        return self.majority, 1.0  # (intent, confidence)


class KeywordBaseline:
    """Hand-written regex rules, first match wins; else 'general_complaint_other'."""
    def fit(self, texts=None, labels=None):
        return self

    def predict(self, text):
        t = text.lower()
        for intent, patterns in KEYWORD_RULES.items():
            for p in patterns:
                if re.search(p, t):
                    return intent, 0.6  # fixed, unconfident-by-design score
        return "general_complaint_other", 0.3


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MLClassifier:
    """TF-IDF + Logistic Regression trained on weakly-labeled bulk data."""
    MODEL_PATH = os.path.join(_ROOT, "data", "ml_classifier.pkl")

    def fit(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000, C=5.0)
        self.clf.fit(X, labels)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump((self.vec, self.clf), f)
        return self

    def load(self):
        with open(self.MODEL_PATH, "rb") as f:
            self.vec, self.clf = pickle.load(f)
        return self

    def predict(self, text):
        X = self.vec.transform([text])
        probs = self.clf.predict_proba(X)[0]
        idx = probs.argmax()
        return self.clf.classes_[idx], float(probs[idx])


class LLMClassifier:
    """Zero-shot intent classification via Anthropic API. Falls back to
    MLClassifier if ANTHROPIC_API_KEY isn't set or the call fails, so the
    pipeline never hard-depends on API access for reproducibility."""
    def __init__(self, fallback: MLClassifier):
        self.fallback = fallback
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def fit(self, texts=None, labels=None):
        return self

    def predict(self, text):
        if not self.enabled:
            return self.fallback.predict(text)
        try:
            import anthropic
            client = anthropic.Anthropic()
            prompt = (
                "Classify this customer support tweet into exactly one intent from this list:\n"
                f"{INTENTS}\n\nTweet: {text!r}\n\n"
                'Respond with ONLY a JSON object: {"intent": "...", "confidence": 0.0-1.0}'
            )
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            out = json.loads(resp.content[0].text.strip())
            return out["intent"], float(out.get("confidence", 0.8))
        except Exception:
            return self.fallback.predict(text)


def is_high_severity(text):
    t = text.lower()
    return any(re.search(p, t) for p in SEVERITY_KEYWORDS)
