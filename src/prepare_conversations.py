"""
Turns the raw tweet table (real Kaggle schema or our synthetic stand-in)
into (customer_opening_message, resolution_reply) conversation pairs.

Logic (same as would be used on the real dataset):
- inbound==True rows with in_response_to_tweet_id == "" are thread openers.
- Walk the response_tweet_id chain to find the LAST outbound (brand) reply
  in that thread -> treat it as the "resolution" reply for grounding.
- Keep the brand's own author_id as the BRAND handle (most frequent
  non-inbound author).
"""
import csv
import json
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(_ROOT, "data", "raw_tweets.csv")
OUT_PATH = os.path.join(_ROOT, "data", "conversations.jsonl")


def load_rows(path=IN_PATH):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_conversations(rows):
    by_id = {r["tweet_id"]: r for r in rows}
    brand_counts = defaultdict(int)
    for r in rows:
        if r["inbound"] == "False":
            brand_counts[r["author_id"]] += 1
    brand = max(brand_counts, key=brand_counts.get)

    openers = [r for r in rows if r["inbound"] == "True" and not r["in_response_to_tweet_id"]]

    convos = []
    for opener in openers:
        chain = [opener]
        cur = opener
        # walk forward via response_tweet_id
        while cur.get("response_tweet_id"):
            nxt = by_id.get(cur["response_tweet_id"])
            if not nxt:
                break
            chain.append(nxt)
            cur = nxt
        last_brand_reply = None
        for msg in reversed(chain):
            if msg["author_id"] == brand:
                last_brand_reply = msg["text"]
                break
        convos.append({
            "conv_id": opener["tweet_id"],
            "customer_message": opener["text"],
            "resolution_reply": last_brand_reply,  # may be None if never resolved in-thread
            "n_turns": len(chain),
            "_gen_intent": opener.get("_gen_intent", ""),
        })
    return convos, brand


if __name__ == "__main__":
    rows = load_rows()
    convos, brand = build_conversations(rows)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in convos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    resolved = sum(1 for c in convos if c["resolution_reply"])
    print(f"brand detected: {brand}")
    print(f"{len(convos)} conversations, {resolved} with an in-thread resolution reply ({resolved/len(convos):.0%})")
