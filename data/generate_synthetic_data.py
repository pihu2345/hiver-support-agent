"""
Generates a synthetic stand-in for the Kaggle "Customer Support on Twitter"
dataset (thoughtvector/customer-support-on-twitter), scoped to ONE brand.

WHY SYNTHETIC: this sandbox's network allowlist does not include kaggle.com
or huggingface.co, so the real dataset could not be downloaded here. This
script produces data with the IDENTICAL column schema as the real dataset
(tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id), so every downstream script in src/ works
unchanged on the real CSV -- just replace data/raw_tweets.csv.

Brand chosen: "BrandHelp" (a stand-in for a support handle like
@AmazonHelp / @AppleSupport / @Delta / @SpotifyCares -- the real dataset
has ~20 such brand handles, this generator models the same *shape*:
short informal customer tweets, an agent asking for DM/order info, a
resolution tweet, occasional escalation to phone/DM, typos, emoji,
multi-turn threads).

Ground-truth intent labels are stored ONLY for generation bookkeeping in
a separate column (`_gen_intent`) that downstream code is NOT allowed to
read for training/eval -- eval/golden_set.csv is built by an independent
manual review pass (see eval/labeling_notes.md) precisely so the eval
isn't circular.
"""
import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

BRAND = "BrandHelp"
BRAND_AUTHOR_ID = "BrandHelp"

# ---- Intent taxonomy (defined FROM the data / from real-world support ontologies) ----
INTENTS = [
    "order_status_delivery",
    "refund_return",
    "product_defect_damaged",
    "billing_charge_dispute",
    "account_login_access",
    "app_website_technical",
    "general_complaint_other",
]

# customer message templates per intent, with slots for messiness
TEMPLATES = {
    "order_status_delivery": [
        "hey {brand} my order #{oid} still hasn't shipped its been {days} days??",
        "@{brand} where is my package? tracking says {days} days late and no updates smh",
        "yo @{brand} order {oid} was supposed to arrive yesterday still nothing",
        "@{brand} any update on order #{oid}? its been stuck in transit forever",
        "why does {brand} tracking say delivered when i never got my order #{oid} !!",
    ],
    "refund_return": [
        "@{brand} i returned item {oid} 2 weeks ago still no refund, whats going on",
        "hi @{brand} can you help me start a return for order #{oid}, wrong size",
        "@{brand} requested a refund on {oid} days ago and nothing, this is ridiculous",
        "how do i return order #{oid} to {brand}? never used the product",
        "@{brand} refund for #{oid} never came through, checked my bank twice",
    ],
    "product_defect_damaged": [
        "@{brand} the item from order #{oid} arrived broken, whole box was crushed",
        "got order {oid} from {brand} today and its defective, doesn't even turn on",
        "@{brand} product {oid} stopped working after 2 days, really disappointed",
        "opened package #{oid} from {brand} and its damaged, packaging was fine but item cracked",
    ],
    "billing_charge_dispute": [
        "@{brand} i was charged twice for order #{oid}, please fix this asap",
        "why is there a charge on my card from {brand} i don't recognize? order #{oid}?",
        "@{brand} billed me the wrong amount for #{oid}, can someone check this",
        "@{brand} unauthorized charge on my account, did not place order #{oid}",
    ],
    "account_login_access": [
        "@{brand} can't log into my account, keeps saying invalid password even after reset",
        "hey @{brand} app won't let me sign in, tried resetting twice",
        "@{brand} locked out of my account for no reason, need this fixed today",
        "@{brand} 2FA code never arrives so i can't access my account",
    ],
    "app_website_technical": [
        "@{brand} app keeps crashing every time i open the checkout page",
        "site is down for anyone else or just me? @{brand}",
        "@{brand} website won't load past the cart page, tried 3 browsers",
        "@{brand} app update broke everything, cant even see my order history now",
    ],
    "general_complaint_other": [
        "@{brand} customer service on the phone was so rude today, unacceptable",
        "honestly @{brand} your support has gone downhill, waited an hour for nothing",
        "@{brand} just a generally bad experience today, thinking of switching brands",
        "@{brand} love the product but the whole ordering experience was a mess",
    ],
}

AGENT_ACK = [
    "Hi there, sorry to hear that! Please DM us your order number and email on file so we can look into this. ^{agent}",
    "We're sorry for the trouble! Send us a DM with your order # and we'll get this sorted right away. ^{agent}",
    "That's not the experience we want for you. Can you DM your order number so we can investigate? ^{agent}",
    "Thanks for flagging this, we'd like to help. Please DM your order # and account email. ^{agent}",
]

AGENT_RESOLUTION = {
    "order_status_delivery": "Thanks for the info! We checked and your order is out for delivery today, tracking has been updated. ^{agent}",
    "refund_return": "We've processed your refund of ${amt} back to your original payment method, please allow 3-5 business days. ^{agent}",
    "product_defect_damaged": "So sorry about that! We've shipped a free replacement and you can keep or discard the damaged item. ^{agent}",
    "billing_charge_dispute": "We found the duplicate charge and issued a refund for ${amt}, you should see it in 3-5 business days. ^{agent}",
    "account_login_access": "We've manually reset your account access, please try logging in again and let us know if it persists. ^{agent}",
    "app_website_technical": "Thanks for reporting this bug! Our engineering team pushed a fix, please update the app and try again. ^{agent}",
    "general_complaint_other": "We're really sorry to hear this and have shared your feedback with the team, we'd love another chance to make it right. ^{agent}",
}

AGENTS = ["JS", "KM", "AR", "TL"]
N_THREADS = 650


def messy(text):
    """add light noise: random casing, dropped punctuation, occasional typo."""
    if random.random() < 0.15:
        text = text.replace("you", "u").replace("please", "pls")
    if random.random() < 0.1:
        text = text[0].lower() + text[1:]
    if random.random() < 0.2:
        text = text + " " + random.choice(["😤", "🙄", "!!", "??", ""])
    return text.strip()


def gen():
    rows = []
    tweet_id = 1
    start = datetime(2024, 1, 1)
    for i in range(N_THREADS):
        intent = random.choices(
            INTENTS,
            weights=[22, 18, 15, 12, 13, 10, 10],  # roughly realistic support mix
        )[0]
        oid = random.randint(100000, 999999)
        days = random.randint(3, 14)
        template = random.choice(TEMPLATES[intent])
        cust_text = messy(template.format(brand=BRAND, oid=oid, days=days))
        cust_id = f"cust_{1000+i}"
        ts0 = start + timedelta(hours=random.randint(0, 6000))

        cust_tweet_id = tweet_id; tweet_id += 1
        ack_tweet_id = tweet_id; tweet_id += 1

        has_followup = random.random() < 0.78
        # ~78% of threads have a customer follow-up (order#) + a real resolution
        # (mirrors real data where the resolving reply is a few hops deep)
        if has_followup:
            followup_id = tweet_id; tweet_id += 1
            resolve_id = tweet_id; tweet_id += 1
        else:
            followup_id = resolve_id = None

        rows.append(dict(
            tweet_id=cust_tweet_id, author_id=cust_id, inbound=True,
            created_at=ts0.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            text=cust_text, response_tweet_id=str(ack_tweet_id),
            in_response_to_tweet_id="", _gen_intent=intent,
        ))

        agent = random.choice(AGENTS)
        ack_text = AGENT_ACK[i % len(AGENT_ACK)].format(agent=agent)
        rows.append(dict(
            tweet_id=ack_tweet_id, author_id=BRAND_AUTHOR_ID, inbound=False,
            created_at=(ts0 + timedelta(minutes=15)).strftime("%a %b %d %H:%M:%S +0000 %Y"),
            text=ack_text, response_tweet_id=str(followup_id) if followup_id else "",
            in_response_to_tweet_id=str(cust_tweet_id), _gen_intent=intent,
        ))

        if has_followup:
            rows.append(dict(
                tweet_id=followup_id, author_id=cust_id, inbound=True,
                created_at=(ts0 + timedelta(minutes=25)).strftime("%a %b %d %H:%M:%S +0000 %Y"),
                text=f"DMd you, order # is {oid}", response_tweet_id=str(resolve_id),
                in_response_to_tweet_id=str(ack_tweet_id), _gen_intent=intent,
            ))
            res_text = AGENT_RESOLUTION[intent].format(agent=agent, amt=f"{random.randint(15,120)}.{random.randint(0,99):02d}")
            rows.append(dict(
                tweet_id=resolve_id, author_id=BRAND_AUTHOR_ID, inbound=False,
                created_at=(ts0 + timedelta(minutes=40)).strftime("%a %b %d %H:%M:%S +0000 %Y"),
                text=res_text, response_tweet_id="",
                in_response_to_tweet_id=str(followup_id), _gen_intent=intent,
            ))

        # ~8% chance of an "escalation-worthy" high-severity variant (legal/fraud/safety language)
        if random.random() < 0.08:
            sev_text = messy(random.choice([
                f"@{BRAND} this is fraud, someone used my card without permission for order #{oid}, calling my bank and lawyer",
                f"@{BRAND} I'm going to report this to the BBB, your product from order {oid} literally caused an injury",
                f"@{BRAND} still no response after 5 days, I want a manager to call me NOW about {oid}",
            ]))
            sev_id = tweet_id; tweet_id += 1
            rows.append(dict(
                tweet_id=sev_id, author_id=cust_id, inbound=True,
                created_at=(ts0 + timedelta(hours=2)).strftime("%a %b %d %H:%M:%S +0000 %Y"),
                text=sev_text, response_tweet_id="",
                in_response_to_tweet_id="", _gen_intent=intent,
            ))

    return rows


if __name__ == "__main__":
    rows = gen()
    fields = ["tweet_id", "author_id", "inbound", "created_at", "text",
              "response_tweet_id", "in_response_to_tweet_id", "_gen_intent"]
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_tweets.csv")
    with open(_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows to data/raw_tweets.csv")
