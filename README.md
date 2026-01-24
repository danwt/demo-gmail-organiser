# Gmail Organiser

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Gmail API](https://img.shields.io/badge/Gmail-API%20v1-red.svg)](https://developers.google.com/gmail/api)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple.svg)](https://openrouter.ai/)

Automatically classify and organize Gmail emails using LLM-powered categorization. Queries Gmail for unlabelled emails, classifies them in batches, applies labels, and archives noise categories.

## How it works

1. Queries Gmail for emails missing all category labels (stateless - no local tracking)
2. Sends batches of 20 emails to an LLM for classification
3. Applies the corresponding Gmail label to each email
4. Archives noise categories (junk, newsletter, purchases, events) out of inbox

Categories are defined in [`taxonomy.yaml`](taxonomy.yaml) and are easy to modify.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:

```
GOOGLE_CLIENT_ID=...        # Google OAuth app credentials
GOOGLE_CLIENT_SECRET=...
OPENROUTER_API_KEY=...      # openrouter.ai API key
OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

Run OAuth (one-time, opens browser):

```bash
python main.py auth
```

## Usage

```bash
python main.py
```

Classifies all emails that don't yet have a category label. Safe to run repeatedly - only processes new/unlabelled emails.

## Docker

```bash
docker compose up
```

Requires `token.json` to exist (run `auth` locally first).

## Taxonomy

Edit `taxonomy.yaml` to add/remove/modify categories:

```yaml
categories:
  - name: junk
    description: any pure spam, scam, marketing, or promotion or other noise
    archive: true
```

Each category becomes a Gmail label. Set `archive: true` to auto-remove from inbox.
