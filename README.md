# Gmail Organiser

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![gogcli](https://img.shields.io/badge/Google-gogcli-red.svg)](https://github.com/steipete/gogcli)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple.svg)](https://openrouter.ai/)

Automatically classify and organize Gmail emails using LLM-powered categorization. Queries Gmail for unlabelled emails, classifies them in batches, applies labels, and archives noise categories.

## How it works

1. Queries Gmail via `gog` for emails missing all category labels (stateless - no local tracking)
2. Sends batches of 20 emails to an LLM for classification
3. Applies the corresponding Gmail label to each email
4. Archives noise categories (junk, newsletter, purchases, events) out of inbox

Categories are defined in [`taxonomy.yaml`](taxonomy.yaml) and are easy to modify.

## Setup

```bash
brew tap steipete/tap
brew install steipete/tap/gogcli
uv sync
```

Copy `.env.example` to `.env` and fill in:

```
GOG_ACCOUNT=danwtisdall     # default account/alias used by this app
GOG_AUTH_EMAIL=...          # real Google email used during auth
GOG_CREDENTIALS=...         # path to OAuth desktop credentials.json
OPENROUTER_API_KEY=...      # openrouter.ai API key
OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

Run OAuth bootstrap (one-time, opens browser):

```bash
uv run python main.py auth
```

Or pass credentials explicitly:

```bash
uv run python main.py auth /absolute/path/to/credentials.json
```

`auth` stores credentials with `gog auth credentials set`, authorizes Gmail+Calendar scopes on `GOG_AUTH_EMAIL`, and optionally sets the `GOG_ACCOUNT` alias.

## Usage

```bash
uv run python main.py run
```

Classifies all emails that don't yet have a category label. Safe to run repeatedly - only processes new/unlabelled emails.

## Docker

```bash
docker compose up
```

The image includes `gog`. Config is persisted in `./.gogcli/`.

## Taxonomy

Edit `taxonomy.yaml` to add/remove/modify categories:

```yaml
categories:
  - name: junk
    description: any pure spam, scam, marketing, or promotion or other noise
    archive: true
```

Each category becomes a Gmail label. Set `archive: true` to auto-remove from inbox.
