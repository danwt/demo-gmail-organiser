import os
import sys
import json
import time
import base64

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
    "junk",
    "newsletter",
    "events",
    "purchases",
    "enquiries",
]

CATEGORY_DESCRIPTIONS = {
    "family-and-friends": "things directly sent to me by family and friends",
    "jobs": "related to jobs I applied for and interview processes etc, does NOT match cold calls/auto reach outs, these are junk",
    "financial": "related to property I own, my stocks, pensions, bank accounts, tax, crypto etc",
    "businesses": "related to businesses I run",
    "cloud": "anything related to cloud infra i'm running on various cloud accounts such as GCP AWS Cloudflare etc",
    "junk": "any pure spam, scam, marketing, or promotion or other noise",
    "newsletter": "any regular newsletter that I signed up for that isn't promotion or marketing, (note includes much discussion on apache org stuff)",
    "events": "anything related to tickets, events or travel plans I actually made such as cinema, holidays, hotel bookings, flights",
    "purchases": "any updates on things I've bought, their delivery, receipts, this includes regular paid subscriptions, note does not apply to TC changes and other noise",
    "enquiries": "any actual enquiries from actual people asking me things who aren't friends or family (not spam or automated)",
}

BATCH_SIZE = 50
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


def extract_body_text(payload):
    if payload.get("mimeType", "").startswith("text/plain"):
        data = payload.get("body", {}).get("data", "")
        if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        text = extract_body_text(part)
        if text:
            return text
    return ""


def fetch_message_metadata(service, msg_id):
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    labels = msg.get("labelIds", [])
    body = extract_body_text(msg.get("payload", {}))
    return {
        "id": msg_id,
        "from": headers.get("From", ""),
        "subject": headers.get("Subject", ""),
        "snippet": msg.get("snippet", ""),
        "body": body[:1000],
        "labels": labels,
    }


def already_classified(msg_metadata, label_map):
    label_ids = set(msg_metadata["labels"])
    category_label_ids = set(label_map[cat] for cat in CATEGORIES)
    return bool(label_ids & category_label_ids)


def classify_batch(llm, messages):
    categories_desc = "\n".join(f"- {cat}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items())
    def format_email(i, m):
        lines = f'{i+1}. From: {m["from"]} | Subject: {m["subject"]}'
        body = m.get("body", "").strip()
        if body:
            lines += f'\n   Body: {body[:500]}'
        elif m.get("snippet"):
            lines += f'\n   Preview: {m["snippet"]}'
        return lines

    emails_desc = "\n".join(format_email(i, m) for i, m in enumerate(messages))

    prompt = f"""Classify each email into exactly ONE of these categories:

{categories_desc}

To be aboslutely clear, the point of this is to filter signal from noise: when in doubt, default to junk!
For example, a cold call or promotion is still junk even if it is promoting crypto or a job (i.e. it's not finance or job categories!)

Emails:
{emails_desc}

Respond with a JSON array of strings, one category per email, in the same order. Example: ["junk", "financial", "jobs"]
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
            content = f.read().strip()
            if content:
                return json.loads(content)
    return []


def save_mappings(mappings):
    with open(MAPPINGS_FILE, "w") as f:
        json.dump(mappings, f, indent=2)


def cmd_auth():
    print("Running OAuth flow...")
    get_gmail_service()
    print("Auth complete, token.json saved.")


def process_batch(service, llm, label_map, msg_ids, mappings, classified_ids, batch_num, force=False):
    batch = []
    category_label_ids = [label_map[cat] for cat in CATEGORIES]
    for msg_id in msg_ids:
        if not force and msg_id in classified_ids:
            continue
        meta = fetch_message_metadata(service, msg_id)
        if not force and already_classified(meta, label_map):
            classified_ids.add(msg_id)
            continue
        if force:
            existing_cat_labels = [lid for lid in meta["labels"] if lid in category_label_ids]
            if existing_cat_labels:
                service.users().messages().modify(
                    userId="me", id=msg_id,
                    body={"removeLabelIds": existing_cat_labels}
                ).execute()
        batch.append(meta)

    if not batch:
        return

    try:
        classifications = classify_batch(llm, batch)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ERROR in batch {batch_num}: {e}, retrying...")
        time.sleep(2)
        try:
            classifications = classify_batch(llm, batch)
        except Exception as e2:
            print(f"  FAILED batch {batch_num}: {e2}, skipping")
            return

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
        classified_ids.add(msg["id"])

    save_mappings(mappings)
    print(f"  Batch {batch_num} done: classified {len(batch)}, total {len(mappings)}")


def cmd_classify():
    force = "--force" in sys.argv
    service = get_gmail_service()
    llm = get_llm_client()

    print("Ensuring labels exist...")
    label_map = ensure_labels(service)

    if force:
        print("Force mode: reclassifying all emails")
        mappings = []
        classified_ids = set()
    else:
        mappings = load_mappings()
        classified_ids = {m["id"] for m in mappings}
        print(f"Already classified: {len(classified_ids)}")

    page_token = None
    batch_num = 0
    total_fetched = 0

    while True:
        resp = service.users().messages().list(
            userId="me", maxResults=BATCH_SIZE, pageToken=page_token
        ).execute()
        msg_ids = [m["id"] for m in resp.get("messages", [])]
        total_fetched += len(msg_ids)
        batch_num += 1

        print(f"  Processing batch {batch_num} ({total_fetched} fetched so far)...")
        process_batch(service, llm, label_map, msg_ids, mappings, classified_ids, batch_num, force)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    print(f"Classification complete: {len(mappings)} emails classified")
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
