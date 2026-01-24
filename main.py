import os
import sys
import json
import time
import base64

import yaml
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
BATCH_SIZE = 20
TAXONOMY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.yaml")


def load_taxonomy():
    with open(TAXONOMY_FILE) as f:
        data = yaml.safe_load(f)
    categories = []
    for cat in data["categories"]:
        categories.append({
            "name": cat["name"],
            "description": cat["description"],
            "archive": cat.get("archive", False),
        })
    return categories


def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_config = {
                "installed": {
                    "client_id": os.environ["GOOGLE_CLIENT_ID"],
                    "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def get_llm_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def ensure_labels(service, taxonomy):
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    label_map = {l["name"]: l["id"] for l in existing}
    for cat in taxonomy:
        if cat["name"] not in label_map:
            created = service.users().labels().create(
                userId="me",
                body={"name": cat["name"], "labelListVisibility": "labelShow", "messageListVisibility": "show"},
            ).execute()
            label_map[cat["name"]] = created["id"]
            print(f"  Created label: {cat['name']}")
    return label_map


def build_unsorted_query(taxonomy):
    return " ".join(f"-label:{cat['name']}" for cat in taxonomy)


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
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    body = extract_body_text(msg.get("payload", {}))
    return {
        "id": msg_id,
        "from": headers.get("From", ""),
        "subject": headers.get("Subject", ""),
        "snippet": msg.get("snippet", ""),
        "body": body[:1000],
    }


def classify_batch(llm, messages, taxonomy):
    category_names = [cat["name"] for cat in taxonomy]
    categories_desc = "\n".join(f"- {cat['name']}: {cat['description']}" for cat in taxonomy)

    def format_email(i, m):
        lines = f'{i+1}. From: {m["from"]} | Subject: {m["subject"]}'
        body = m.get("body", "").strip()
        if body:
            lines += f"\n   Body: {body[:500]}"
        elif m.get("snippet"):
            lines += f"\n   Preview: {m['snippet']}"
        return lines

    emails_desc = "\n".join(format_email(i, m) for i, m in enumerate(messages))

    prompt = f"""Classify each email into exactly ONE of these categories:

{categories_desc}

To be absolutely clear, the point of this is to filter signal from noise: when in doubt, default to junk!
For example, a cold call or promotion is still junk even if it is promoting crypto or a job (i.e. it's not finance or job categories!)

Emails:
{emails_desc}

Respond with a JSON array of strings, one category per email, in the same order. Example: ["junk", "financial", "jobs"]
Only use category names from the list above. Respond with ONLY the JSON array, no other text."""

    resp = llm.chat.completions.create(
        model=os.environ["OPENROUTER_MODEL"],
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
        if c not in category_names:
            raise ValueError(f"Invalid category: {c}")
    return classifications


def process_batch(service, llm, label_map, taxonomy, msg_ids, batch_num):
    archive_categories = {cat["name"] for cat in taxonomy if cat["archive"]}
    batch = []
    for msg_id in msg_ids:
        meta = fetch_message_metadata(service, msg_id)
        batch.append(meta)

    if not batch:
        return 0

    try:
        classifications = classify_batch(llm, batch, taxonomy)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Batch {batch_num} classification error: {e}, retrying...")
        time.sleep(2)
        try:
            classifications = classify_batch(llm, batch, taxonomy)
        except Exception as e2:
            print(f"  Batch {batch_num} failed: {e2}, skipping")
            return 0

    classified = 0
    for msg, cat in zip(batch, classifications):
        modify_body = {"addLabelIds": [label_map[cat]]}
        if cat in archive_categories:
            modify_body["removeLabelIds"] = ["INBOX"]
        try:
            service.users().messages().modify(userId="me", id=msg["id"], body=modify_body).execute()
            classified += 1
        except HttpError:
            continue

    print(f"  Batch {batch_num}: classified {classified} emails")
    return classified


def run():
    taxonomy = load_taxonomy()
    service = get_gmail_service()
    llm = get_llm_client()

    label_map = ensure_labels(service, taxonomy)
    query = build_unsorted_query(taxonomy)
    print(f"Querying for unsorted emails...")

    page_token = None
    batch_num = 0
    total_classified = 0

    while True:
        resp = service.users().messages().list(
            userId="me", q=query, maxResults=BATCH_SIZE, pageToken=page_token,
        ).execute()

        msg_ids = [m["id"] for m in resp.get("messages", [])]
        if not msg_ids:
            break

        batch_num += 1
        total_classified += process_batch(service, llm, label_map, taxonomy, msg_ids, batch_num)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    print(f"Done: classified {total_classified} emails")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "auth":
        get_gmail_service()
        print("Auth complete, token.json saved.")
    elif cmd == "run":
        run()
    else:
        print(f"Usage: python main.py [auth|run]")
        sys.exit(1)
