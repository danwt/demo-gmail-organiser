import os
import sys
import json
import time

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from openai import OpenAI

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.environ["OPENROUTER_MODEL"]

CATEGORIES = [
    "family-and-friends",
    "jobs",
    "financial",
    "businesses",
    "cloud",
    "spam",
    "newsletter",
    "events",
    "purchases",
    "enquiries",
]

CATEGORY_DESCRIPTIONS = {
    "family-and-friends": "things directly sent to me by family and friends",
    "jobs": "related to jobs I applied for and interview processes etc",
    "financial": "related to property I own, my stocks, pensions, bank accounts, tax, crypto etc",
    "businesses": "related to businesses I run",
    "cloud": "anything related to cloud infra i'm running on various cloud accounts such as GCP AWS Cloudflare etc",
    "spam": "any pure spam or scam or marketing or promotion",
    "newsletter": "any regular newsletter that I signed up for that isn't promotion or marketing",
    "events": "anything related to tickets, events or travel plans I actually made such as cinema, holidays, hotel bookings, flights",
    "purchases": "any updates on things I've bought, their delivery, receipts, this includes regular paid subscriptions",
    "enquiries": "any actual enquiries from actual people asking me things who aren't friends or family (not spam or automated)",
}

BATCH_SIZE = 30
MAPPINGS_FILE = "mappings.json"


def get_client_config():
    return {
        "installed": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(get_client_config(), SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def get_llm_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def ensure_labels(service):
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    label_map = {l["name"]: l["id"] for l in existing}
    for cat in CATEGORIES:
        if cat not in label_map:
            created = service.users().labels().create(
                userId="me", body={"name": cat, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
            ).execute()
            label_map[cat] = created["id"]
            print(f"  Created label: {cat}")
    return label_map


def fetch_all_message_ids(service):
    ids = []
    page_token = None
    while True:
        resp = service.users().messages().list(
            userId="me", maxResults=500, pageToken=page_token
        ).execute()
        ids.extend(msg["id"] for msg in resp.get("messages", []))
        page_token = resp.get("nextPageToken")
        print(f"  Fetched {len(ids)} message IDs...")
        if not page_token:
            break
    return ids


def fetch_message_metadata(service, msg_id):
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="metadata",
        metadataHeaders=["From", "Subject"]
    ).execute()
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    labels = msg.get("labelIds", [])
    return {
        "id": msg_id,
        "from": headers.get("From", ""),
        "subject": headers.get("Subject", ""),
        "snippet": msg.get("snippet", ""),
        "labels": labels,
    }


def already_classified(msg_metadata, label_map):
    label_ids = set(msg_metadata["labels"])
    category_label_ids = set(label_map[cat] for cat in CATEGORIES)
    return bool(label_ids & category_label_ids)


def classify_batch(llm, messages):
    categories_desc = "\n".join(f"- {cat}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items())
    emails_desc = "\n".join(
        f'{i+1}. From: {m["from"]} | Subject: {m["subject"]} | Preview: {m["snippet"][:100]}'
        for i, m in enumerate(messages)
    )

    prompt = f"""Classify each email into exactly ONE of these categories:

{categories_desc}

Emails:
{emails_desc}

Respond with a JSON array of strings, one category per email, in the same order. Example: ["spam", "financial", "jobs"]
Only use category names from the list above. Respond with ONLY the JSON array, no other text."""

    resp = llm.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    classifications = json.loads(raw)

    if len(classifications) != len(messages):
        raise ValueError(f"Expected {len(messages)} classifications, got {len(classifications)}")
    for c in classifications:
        if c not in CATEGORIES:
            raise ValueError(f"Invalid category: {c}")
    return classifications


def load_mappings():
    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE) as f:
            return json.load(f)
    return []


def save_mappings(mappings):
    with open(MAPPINGS_FILE, "w") as f:
        json.dump(mappings, f, indent=2)


def cmd_auth():
    print("Running OAuth flow...")
    get_gmail_service()
    print("Auth complete, token.json saved.")


def cmd_classify():
    service = get_gmail_service()
    llm = get_llm_client()

    print("Ensuring labels exist...")
    label_map = ensure_labels(service)

    print("Fetching all message IDs...")
    all_ids = fetch_all_message_ids(service)
    print(f"Total messages: {len(all_ids)}")

    mappings = load_mappings()
    classified_ids = {m["id"] for m in mappings}

    to_classify = []
    print("Fetching metadata and filtering already-classified...")
    for i, msg_id in enumerate(all_ids):
        if msg_id in classified_ids:
            continue
        meta = fetch_message_metadata(service, msg_id)
        if already_classified(meta, label_map):
            continue
        to_classify.append(meta)
        if (i + 1) % 100 == 0:
            print(f"  Checked {i+1}/{len(all_ids)}, queued {len(to_classify)} for classification")
        time.sleep(0.05)

    print(f"Emails to classify: {len(to_classify)}")

    for batch_start in range(0, len(to_classify), BATCH_SIZE):
        batch = to_classify[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(to_classify) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Classifying batch {batch_num}/{total_batches}...")

        try:
            classifications = classify_batch(llm, batch)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ERROR in batch {batch_num}: {e}, retrying...")
            time.sleep(2)
            try:
                classifications = classify_batch(llm, batch)
            except Exception as e2:
                print(f"  FAILED batch {batch_num}: {e2}, skipping")
                continue

        for msg, cat in zip(batch, classifications):
            service.users().messages().modify(
                userId="me", id=msg["id"],
                body={"addLabelIds": [label_map[cat]]}
            ).execute()
            mappings.append({
                "id": msg["id"],
                "from": msg["from"],
                "subject": msg["subject"],
                "category": cat,
            })
            time.sleep(0.02)

        save_mappings(mappings)
        print(f"  Batch {batch_num} done, {len(mappings)} total classified")

    print(f"Classification complete. {len(mappings)} emails classified.")
    print(f"Mappings saved to {MAPPINGS_FILE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <auth|classify>")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "auth":
        cmd_auth()
    elif cmd == "classify":
        cmd_classify()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
