import base64
import json
import os
import shlex
import subprocess
import sys
import time

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BATCH_SIZE = 20
BODY_LIMIT = 4000
TAXONOMY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.yaml")
GOG_BIN = os.environ.get("GOG_BIN", "gog")
GOG_ACCOUNT = os.environ.get("GOG_ACCOUNT", "danwtisdall")
GOG_AUTH_EMAIL = os.environ.get("GOG_AUTH_EMAIL", GOG_ACCOUNT)
GOG_CREDENTIALS = os.environ.get("GOG_CREDENTIALS")
SEARCH_PAGE_SIZE = 100


def run_gog(args, *, include_account=True, expect_json=False):
    cmd = [GOG_BIN]
    if include_account:
        cmd.extend(["--account", GOG_ACCOUNT])
    if expect_json:
        cmd.extend(["--json", "--results-only"])
    cmd.extend(args)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"gog binary not found: {GOG_BIN}") from exc
    except subprocess.CalledProcessError as exc:
        quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or f"exit={exc.returncode}"
        raise RuntimeError(f"command failed: {quoted_cmd}\n{details}") from exc

    output = result.stdout.strip()
    if not expect_json:
        return output
    if not output:
        return None

    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON from gog: {output[:500]}") from exc


def as_items(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("messages", "threads", "items", "events", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        result = payload.get("result")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("messages", "threads", "items", "events", "results", "data"):
                value = result.get(key)
                if isinstance(value, list):
                    return value
    return []


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


def get_llm_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def ensure_labels(taxonomy):
    existing = as_items(run_gog(["gmail", "labels", "list"], expect_json=True))
    label_map = {}
    for label in existing:
        if not isinstance(label, dict):
            continue
        name = label.get("name")
        label_id = label.get("id")
        if isinstance(name, str) and name:
            label_map[name] = label_id

    for cat in taxonomy:
        if cat["name"] not in label_map:
            created = run_gog(["gmail", "labels", "create", cat["name"]], expect_json=True)
            label_map[cat["name"]] = created.get("id") if isinstance(created, dict) else None
            print(f"  Created label: {cat['name']}")

    return label_map


def build_unsorted_query(taxonomy):
    return " ".join(f"-label:{cat['name']}" for cat in taxonomy)


def decode_base64_text(data):
    if not isinstance(data, str) or not data:
        return ""
    padding = (-len(data)) % 4
    if padding:
        data += "=" * padding
    try:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_body_text(payload):
    if not isinstance(payload, dict):
        return ""

    body_data = payload.get("body", {}).get("data")
    decoded = decode_base64_text(body_data)
    if decoded:
        return decoded

    if payload.get("mimeType", "").startswith("text/plain"):
        plain_data = payload.get("body", {}).get("data", "")
        plain_decoded = decode_base64_text(plain_data)
        if plain_decoded:
            return plain_decoded

    for part in payload.get("parts", []):
        text = extract_body_text(part)
        if text:
            return text

    return ""


def get_header(headers, name):
    wanted = name.lower()
    if isinstance(headers, dict):
        for key, value in headers.items():
            if str(key).lower() == wanted:
                return str(value)
        return ""
    if isinstance(headers, list):
        for entry in headers:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("name", "")).lower() == wanted:
                return str(entry.get("value", ""))
    return ""


def normalize_message(raw):
    if not isinstance(raw, dict):
        return None

    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    headers = raw.get("headers") or payload.get("headers") or {}

    body = ""
    for key in ("body", "bodyText", "text"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            body = value
            break
    if not body:
        body = extract_body_text(payload)

    msg_id = raw.get("id") or raw.get("messageId") or raw.get("message_id")
    if not isinstance(msg_id, str) or not msg_id:
        return None

    return {
        "id": msg_id,
        "thread_id": raw.get("threadId") or raw.get("thread_id") or "",
        "from": raw.get("from") or get_header(headers, "From"),
        "subject": raw.get("subject") or get_header(headers, "Subject"),
        "snippet": raw.get("snippet") or raw.get("preview") or "",
        "body": body[:BODY_LIMIT] if isinstance(body, str) else "",
    }


def classify_batch(llm, messages, taxonomy):
    category_names = [cat["name"] for cat in taxonomy]
    categories_desc = "\n".join(f"- {cat['name']}: {cat['description']}" for cat in taxonomy)

    def format_email(i, m):
        lines = f'{i+1}. From: {m["from"]} | Subject: {m["subject"]}'
        body = m.get("body", "").strip()
        if body:
            lines += f"\n   Body: {body[:BODY_LIMIT]}"
        elif m.get("snippet"):
            lines += f"\n   Preview: {m['snippet']}"
        return lines

    emails_desc = "\n".join(format_email(i, m) for i, m in enumerate(messages))

    prompt = f"""Classify each email into exactly ONE of these categories:

{categories_desc}

To be absolutely clear, the point of this is to filter signal from noise: when in doubt, default to junk!

Specific traps to watch for:
- AliExpress / Amazon / Etsy / eBay / Shein / Temu and similar marketplaces: these senders blast both real order/shipping/delivery/receipt updates AND constant marketing/promo/recommendation/discount blasts. Read the body carefully. Only use "purchases" if the email is genuinely about an order I actually placed (order confirmation, shipping update, delivery, refund, receipt). If it's "deals you'll love", "recommended for you", "X% off", "flash sale", "items in your wishlist are on sale", price-drops on browsed items, or any generic promo — that is "junk", not "purchases".
- LinkedIn: distinguish real human messages directed at me (a real person sending me a message, an interview-process update, a recruiter from a company I actually applied to) from LinkedIn's automated noise (job alerts, "people you may know", "X viewed your profile", "trending in your network", weekly digests, posts from companies, sponsored InMail / cold recruiter spam from people I have no relationship with). Real targeted human messages → "jobs" or "enquiries" as appropriate. Automated LinkedIn noise and cold recruiter spam → "junk".
- Generic "your account / terms have been updated" / "new privacy policy" / "we've made changes" emails from services I use → "junk", not "purchases" or "cloud".
- Cloud provider (GCP/AWS/Cloudflare/Vercel/etc) marketing, webinars, "what's new" newsletters → "junk" or "newsletter", NOT "cloud". Use "cloud" only for actual billing, usage alerts, security warnings, or operational notices on accounts I run.

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


def apply_labels(message_ids, category, should_archive):
    if not message_ids:
        return 0

    cmd = ["gmail", "batch", "modify", *message_ids, f"--add={category}"]
    if should_archive:
        cmd.append("--remove=INBOX")

    run_gog(cmd)
    return len(message_ids)


def fetch_messages(query):
    result = run_gog(
        [
            "gmail",
            "messages",
            "search",
            "--all",
            "--max",
            str(SEARCH_PAGE_SIZE),
            "--include-body",
            "--",
            query,
        ],
        expect_json=True,
    )

    messages = []
    seen_ids = set()
    for entry in as_items(result):
        normalized = normalize_message(entry)
        if not normalized:
            continue
        msg_id = normalized["id"]
        if msg_id in seen_ids:
            continue
        seen_ids.add(msg_id)
        messages.append(normalized)
    return messages


def strip_category_labels(taxonomy, message_ids):
    if not message_ids:
        return
    chunk_size = 500
    remove_flags = [f"--remove={cat['name']}" for cat in taxonomy]
    for i in range(0, len(message_ids), chunk_size):
        chunk = message_ids[i:i + chunk_size]
        run_gog(["gmail", "batch", "modify", *chunk, *remove_flags])


def classify_and_apply(llm, taxonomy, messages):
    if not messages:
        return 0
    archive_categories = {cat["name"] for cat in taxonomy if cat["archive"]}
    total_classified = 0
    batch_num = 0
    for i in range(0, len(messages), BATCH_SIZE):
        batch_messages = messages[i:i + BATCH_SIZE]
        batch_num += 1
        try:
            classifications = classify_batch(llm, batch_messages, taxonomy)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Batch {batch_num} classification error: {e}, retrying...")
            time.sleep(2)
            try:
                classifications = classify_batch(llm, batch_messages, taxonomy)
            except Exception as e2:
                print(f"  Batch {batch_num} failed: {e2}, skipping")
                continue

        by_category = {}
        for message, category in zip(batch_messages, classifications):
            by_category.setdefault(category, []).append(message["id"])

        batch_classified = 0
        for category, message_ids in by_category.items():
            try:
                batch_classified += apply_labels(
                    message_ids,
                    category,
                    should_archive=category in archive_categories,
                )
            except RuntimeError as exc:
                print(f"  Batch {batch_num} apply error for category {category}: {exc}")

        total_classified += batch_classified
        print(f"  Batch {batch_num}: classified {batch_classified} emails")
    return total_classified


def run():
    taxonomy = load_taxonomy()
    llm = get_llm_client()

    ensure_labels(taxonomy)
    query = build_unsorted_query(taxonomy)
    print("Querying for unsorted emails...")

    messages = fetch_messages(query)
    if not messages:
        print("Done: classified 0 emails")
        return

    print(f"Found {len(messages)} candidate emails")
    total_classified = classify_and_apply(llm, taxonomy, messages)
    print(f"Done: classified {total_classified} emails")


def relabel(since_date):
    gmail_date = since_date.replace("-", "/")

    taxonomy = load_taxonomy()
    llm = get_llm_client()
    ensure_labels(taxonomy)

    print(f"Querying for all emails since {gmail_date}...")
    messages = fetch_messages(f"after:{gmail_date}")
    if not messages:
        print("Done: classified 0 emails")
        return

    print(f"Found {len(messages)} emails since {gmail_date}")
    print("Stripping existing category labels...")
    strip_category_labels(taxonomy, [m["id"] for m in messages])

    total_classified = classify_and_apply(llm, taxonomy, messages)
    print(f"Done: relabelled {total_classified} emails")


def auth(credentials_path=None):
    if "@" not in GOG_AUTH_EMAIL:
        raise RuntimeError(
            "GOG_AUTH_EMAIL must be a full email address for auth flows."
        )

    if credentials_path:
        run_gog(["auth", "credentials", "set", credentials_path], include_account=False)

    run_gog(
        ["auth", "add", GOG_AUTH_EMAIL, "--services", "gmail,calendar"],
        include_account=False,
    )

    if GOG_ACCOUNT != GOG_AUTH_EMAIL:
        run_gog(
            ["auth", "alias", "set", GOG_ACCOUNT, GOG_AUTH_EMAIL],
            include_account=False,
        )

    print(f"Auth complete for account {GOG_AUTH_EMAIL}.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "auth":
        credentials_arg = sys.argv[2] if len(sys.argv) > 2 else GOG_CREDENTIALS
        auth(credentials_arg)
    elif cmd == "run":
        run()
    elif cmd == "relabel":
        if len(sys.argv) < 3:
            print("Usage: python main.py relabel YYYY-MM-DD")
            sys.exit(1)
        relabel(sys.argv[2])
    else:
        print("Usage: python main.py [auth [credentials.json]|run|relabel YYYY-MM-DD]")
        sys.exit(1)
