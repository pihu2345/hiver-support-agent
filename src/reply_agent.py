"""
The support agent pipeline: one function in -> full decision out.

  classify()  -> intent + confidence
  retrieve()  -> most similar historically-resolved case for that intent
  draft()     -> a reply grounded in that historical resolution
  decide()    -> auto_handle vs escalate_to_human, with a stated reason

Escalation policy (deliberately simple & auditable -- a human should be
able to read this function and understand every decision):
  1. High-severity language (fraud/legal/safety/BBB/lawyer/etc.) -> ESCALATE.
     Reason: liability + brand-risk categories should never be auto-replied to.
  2. Classifier confidence below CONF_THRESHOLD -> ESCALATE.
     Reason: don't auto-send a reply grounded in the wrong intent bucket.
  3. Predicted intent is billing_charge_dispute AND text mentions
     "unauthorized" -> ESCALATE regardless of confidence.
     Reason: possible fraud, needs a human + verification, not a template.
  4. Otherwise -> AUTO_HANDLE.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from intents import MLClassifier, LLMClassifier, is_high_severity
from playbook import Playbook

CONF_THRESHOLD = 0.55


def load_pipeline(exclude_ids=None):
    ml = MLClassifier().load() if os.path.exists(MLClassifier.MODEL_PATH) else None
    if ml is None:
        raise RuntimeError("Run `python src/train_classifier.py` first.")
    classifier = LLMClassifier(fallback=ml)
    pb = Playbook().build(exclude_ids=exclude_ids)
    return classifier, pb


def decide_escalation(intent, confidence, text):
    if is_high_severity(text):
        return True, "high-severity language detected (fraud/legal/safety/BBB) -- liability risk, needs a human"
    if confidence < CONF_THRESHOLD:
        return True, f"low classifier confidence ({confidence:.2f} < {CONF_THRESHOLD}) -- unsure which playbook applies"
    if intent == "billing_charge_dispute" and "unauthorized" in text.lower():
        return True, "possible fraud/unauthorized charge -- requires identity verification, not a template reply"
    return False, "routine, high-confidence match to a known resolvable pattern"


def draft_reply(intent, text, retrieved):
    """Grounded reply drafting. If ANTHROPIC_API_KEY is set, ask the LLM to
    adapt the retrieved historical resolution to this specific message.
    Otherwise, fall back to a deterministic template built FROM the
    retrieved historical reply (still grounded, just less fluent)."""
    if not retrieved:
        return ("Thanks for reaching out -- could you DM us your order number "
                "and the email on your account so we can look into this?")
    (hist_customer_msg, hist_reply), sim = retrieved[0]

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            prompt = (
                "You are a customer support agent. A customer sent this message:\n"
                f"\"{text}\"\n\n"
                "Here is how a real agent resolved a similar past case:\n"
                f"\"{hist_reply}\"\n\n"
                "Write a short, on-brand reply for THIS customer, grounded in that "
                "past resolution pattern. Do not invent order numbers, refund "
                "amounts, or promises not implied by the past resolution. "
                "Keep it under 280 characters, one message, no hashtags."
            )
            resp = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception:
            pass  # fall through to template

    # Deterministic, grounded fallback (no LLM call needed to run the demo)
    return hist_reply


def handle_message(text, classifier, pb):
    intent, conf = classifier.predict(text)
    retrieved = pb.retrieve(intent, text, k=1)
    escalate, reason = decide_escalation(intent, conf, text)
    reply = None if escalate else draft_reply(intent, text, retrieved)
    return {
        "text": text,
        "predicted_intent": intent,
        "confidence": round(conf, 3),
        "action": "escalate_to_human" if escalate else "auto_handle",
        "reason": reason,
        "grounding_source": retrieved[0][0][1] if retrieved else None,
        "grounding_similarity": round(retrieved[0][1], 3) if retrieved else None,
        "draft_reply": reply,
    }


if __name__ == "__main__":
    classifier, pb = load_pipeline()
    demo_msgs = [
        "@BrandHelp order #555123 still hasn't shipped, its been 9 days",
        "@BrandHelp this is fraud, unauthorized charge on my card, calling my lawyer",
        "@BrandHelp app keeps crashing on checkout, so annoying",
    ]
    for m in demo_msgs:
        print(handle_message(m, classifier, pb))
        print("---")
