"""
Trains MLClassifier on ALL conversations EXCEPT the golden eval set
(no train/test leakage), using the weak `_gen_intent` label as training
signal. This mirrors a realistic bootstrap: cheap noisy labels to train
on, small hand-reviewed set to actually trust your metrics.
"""
import json
import csv
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from intents import MLClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_PATH = os.path.join(_ROOT, "data", "conversations.jsonl")
GOLDEN_PATH = os.path.join(_ROOT, "eval", "golden_set.csv")


def main():
    golden_ids = {r["id"] for r in csv.DictReader(open(GOLDEN_PATH, encoding="utf-8"))}
    convos = [json.loads(l) for l in open(CONV_PATH, encoding="utf-8")]
    train = [c for c in convos if c["conv_id"] not in golden_ids]
    texts = [c["customer_message"] for c in train]
    labels = [c["_gen_intent"] for c in train]
    MLClassifier().fit(texts, labels)
    print(f"trained on {len(texts)} examples (excluded {len(golden_ids)} golden IDs), "
          f"saved to {MLClassifier.MODEL_PATH}")


if __name__ == "__main__":
    main()
