# News Collector

A lightweight internal tool that lets team members submit news items to a newsletter pipeline through a conversational chat interface. The LLM guides each contributor through providing a source URL, their reason for adding it, and — crucially — a short agreed newsletter blurb that's ready to go.

## How it works

1. A contributor visits the chat page
2. They're greeted by the assistant, which asks for their name
3. They paste a URL and explain why it's worth including
4. The assistant reflects their reasoning back, asks a follow-up question to sharpen the angle, then drafts a punchy 2–4 sentence newsletter blurb
5. The contributor iterates on the blurb until they're happy with it
6. Once confirmed, the item is saved automatically
7. At the end of the week or month, the feed page shows all agreed blurbs — nearly ready to drop into the newsletter

**The "reason" field is kept privately for audit purposes.** Only the agreed blurb appears on the feed page.

---

## Setup

### 1. Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai) API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set your OPENROUTER_API_KEY (and any other values you want to change)
```

### 4. Run

```bash
python app.py
```

The server starts on `http://0.0.0.0:8000` by default.

- **Chat interface**: `http://your-server:8000/`
- **Feed / pipeline view**: `http://your-server:8000/feed`
- **Raw JSON feed**: `http://your-server:8000/api/feed`

---

## File structure

```
news_collector/
├── app.py           Main FastAPI application — routes, session management, tag parsing
├── db.py            SQLite operations (init, save, query)
├── llm.py           OpenRouter client + system prompt
├── config.py        Settings loaded from .env
├── templates/
│   ├── chat.html    Chat interface
│   └── feed.html    Feed / pipeline browser
├── requirements.txt
├── .env.example     Copy this to .env
└── README.md
```

---

## Database schema

```sql
CREATE TABLE news_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    submitter_name  TEXT    NOT NULL,   -- who added it
    url             TEXT    NOT NULL,   -- link to source content
    reason          TEXT    NOT NULL,   -- private audit note (not published)
    agreed_text     TEXT    NOT NULL,   -- the agreed newsletter blurb
    created_at      TEXT    NOT NULL    -- ISO-8601 UTC timestamp
);
```

The SQLite file is created automatically on first run at the path set in `DB_PATH` (default: `news.db` in the working directory).

---

## Changing the model

Edit `MODEL` in your `.env` file. Any model available on OpenRouter works. Recommended:

| Model | Notes |
|---|---|
| `anthropic/claude-3-5-haiku-20241022` | Fast and cheap — good default |
| `anthropic/claude-3-5-sonnet-20241022` | Better at nuanced editorial drafts |
| `openai/gpt-4o-mini` | Good OpenAI alternative |

---

## Running behind a reverse proxy (nginx / Caddy)

For production use on a private server, put the app behind nginx or Caddy with HTTPS. Example nginx config:

```nginx
server {
    listen 443 ssl;
    server_name news.yourserver.internal;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

Then run the app with `HOST=127.0.0.1` in your `.env` so it only listens locally.

---

## Running as a service (systemd)

```ini
# /etc/systemd/system/news-collector.service
[Unit]
Description=News Collector
After=network.target

[Service]
WorkingDirectory=/path/to/news_collector
EnvironmentFile=/path/to/news_collector/.env
ExecStart=/usr/bin/python3 app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now news-collector
```
