from __future__ import annotations

"""
app.py — FastAPI application: routes, session management, tag parsing.

Routes
------
GET  /              Chat interface (HTML)
GET  /feed          Browse all saved news items (HTML)
GET  /api/new-session   Create a session and get an initial greeting
POST /api/chat          Send a message, get a response
GET  /api/feed          JSON list of all saved items

Session management
------------------
Conversation histories are stored in-memory in a plain dict keyed by a UUID
session ID. The session ID lives in the browser's localStorage and is sent
with every chat request. This is deliberately simple — sessions are lost on
server restart, which is fine for a private tool.

Tag parsing
-----------
When the LLM has collected and the user has agreed on a news item, the model
embeds a <SAVE_ITEM>...</SAVE_ITEM> block in its response. This module:
  1. Detects that block with a regex
  2. Extracts the four fields (name, url, reason, agreed_text)
  3. Saves them to SQLite
  4. Strips the raw XML from the displayed response
  5. Returns the saved item metadata to the frontend for a confirmation flash

Feed injection
--------------
If the user's message looks like a request to see the feed, the current items
are fetched from SQLite and injected into the conversation as a system-style
assistant message before the LLM is called. The LLM can then summarise the
feed naturally in its reply.
"""

import json
import re
import time
import uuid
import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import settings
from db import get_feed, get_item_count, get_items_by_name, init_db, mark_done, save_news_item, update_news_item
from llm import chat as llm_chat

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title=settings.app_title)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory session store: { session_id: [{"role": ..., "content": ...}, ...] }
sessions: dict[str, list[dict]] = {}

MAX_SESSIONS = 500          # prune oldest when exceeded
MAX_HISTORY = 60            # messages per session before trimming

# Keywords that trigger feed injection into the LLM context
FEED_KEYWORDS = (
    "feed", "recent", "latest", "what's been added", "what has been added",
    "show me", "show items", "see items", "submitted", "in the pipeline",
    "so far", "what's there", "what is there", "list", "entries",
)

# Keywords that indicate the user wants to update/edit a previous item
UPDATE_KEYWORDS = (
    "update", "edit", "change", "amend", "correct", "fix", "revise",
    "i made a mistake", "wrong", "update my", "edit my", "change my",
)

# ---------------------------------------------------------------------------
# Rate limiting (in-memory, no extra dependencies)
# ---------------------------------------------------------------------------

_rate_limit: dict[str, list] = {}   # { ip: [count, window_start] }
_RL_MAX    = 20     # max requests per window
_RL_WINDOW = 60.0   # window length in seconds


def _check_rate_limit(ip: str) -> bool:
    """Return True if the request is within the allowed rate, False if over limit."""
    now = time.monotonic()
    if ip not in _rate_limit:
        _rate_limit[ip] = [1, now]
        return True
    count, start = _rate_limit[ip]
    if now - start > _RL_WINDOW:
        # Window has expired — start a fresh one
        _rate_limit[ip] = [1, now]
        return True
    if count >= _RL_MAX:
        return False
    _rate_limit[ip][0] += 1
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_session(session_id: str) -> list[dict]:
    """Return the history list for a session, creating it if needed."""
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            # Drop the oldest 10% of sessions
            evict = list(sessions.keys())[: MAX_SESSIONS // 10]
            for k in evict:
                del sessions[k]
        sessions[session_id] = []
    return sessions[session_id]


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep the conversation within MAX_HISTORY messages."""
    if len(history) > MAX_HISTORY:
        # Always keep the first message (usually the greeting) and the tail
        return history[:1] + history[-(MAX_HISTORY - 1):]
    return history


def _wants_feed(message: str) -> bool:
    """Return True if the user's message is asking to see the feed."""
    lower = message.lower()
    return any(kw in lower for kw in FEED_KEYWORDS)


def _build_feed_context(limit: int = 15) -> str:
    """Return a text block describing recent items, to inject into LLM context."""
    items = get_feed(limit=limit)
    if not items:
        return "FEED DATA: No items have been saved yet."

    lines = [f"FEED DATA — {len(items)} most recent item(s):"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"\n[{i}] Added by {item['submitter_name']} on {item['created_at'][:10]}\n"
            f"    URL: {item['url']}\n"
            f"    Blurb: {item['agreed_text']}"
        )
    return "\n".join(lines)


def _extract_name_from_history(history: list[dict]) -> str | None:
    """
    Try to find the contributor's name from the conversation history.
    Looks for the name the LLM has been using — typically found in prior
    assistant messages that addressed the user by name.
    Falls back to None if it cannot be determined.
    """
    # Scan assistant messages for "Hi [Name]", "Thanks [Name]", etc.
    # The LLM consistently uses the contributor's name in replies.
    # A simple heuristic: look for the most recent assistant message that
    # contains a capitalised word following a greeting pattern.
    import re as _re
    greeting_pattern = _re.compile(
        r"(?:hi|hello|thanks|thank you|great|perfect|sure|ok|okay)[,!]?\s+([A-Z][a-z]+)",
        _re.IGNORECASE,
    )
    # Walk history in reverse to find the most recent name usage
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            m = greeting_pattern.search(msg.get("content", ""))
            if m:
                return m.group(1)
    return None


def _wants_update(message: str) -> bool:
    """Return True if the user's message suggests they want to edit a previous item."""
    lower = message.lower()
    return any(kw in lower for kw in UPDATE_KEYWORDS)


def _build_items_context(name: str) -> str:
    """Return a text block listing recent items by this contributor, to inject into LLM context."""
    items = get_items_by_name(name, limit=10)
    if not items:
        return f"CONTRIBUTOR ITEMS: No items found for '{name}'."

    lines = [f"CONTRIBUTOR ITEMS — recent items submitted by {name}:"]
    for item in items:
        status = "done" if item["done"] else "pending"
        lines.append(
            f"\n  ID {item['id']} ({status}) — added {item['created_at'][:10]}\n"
            f"  URL: {item['url']}\n"
            f"  Blurb: {item['agreed_text'][:120]}{'…' if len(item['agreed_text']) > 120 else ''}"
        )
    return "\n".join(lines)


def _parse_update_tag(response_text: str) -> tuple[str, dict | None]:
    """
    Look for an <UPDATE_ITEM>...</UPDATE_ITEM> block in the LLM response.

    Returns
    -------
    (cleaned_text, updated_item | None)
      cleaned_text  : response with the XML block stripped out
      updated_item  : the dict returned by db.update_news_item, or None if no tag found
    """
    pattern = re.compile(r"<UPDATE_ITEM>(.*?)</UPDATE_ITEM>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(response_text)
    if not match:
        return response_text, None

    block = match.group(1)

    def extract(tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    item_id_str    = extract("id")
    name           = extract("name")
    url            = extract("url")
    reason         = extract("reason")
    agreed_text    = extract("agreed_text")

    if not all([item_id_str, name, url, reason, agreed_text]):
        cleaned = pattern.sub("", response_text).strip()
        return cleaned, None

    try:
        item_id = int(item_id_str)
    except ValueError:
        cleaned = pattern.sub("", response_text).strip()
        return cleaned, None

    updated = update_news_item(
        item_id=item_id,
        submitter_name=name,
        url=url,
        reason=reason,
        agreed_text=agreed_text,
    )

    cleaned = pattern.sub("", response_text).strip()
    return cleaned, updated


def _strip_dashes(text: str) -> str:
    """
    Remove em dashes and en-dashes used as em dashes from model output.
    Replaces ' — ' and ' – ' (spaced) with ', ' and unspaced variants with '-'.
    """
    # Spaced em dash → comma-space (reads most naturally in running prose)
    text = re.sub(r'\s*—\s*', ', ', text)
    # Spaced en dash used as a clause separator → comma-space
    text = re.sub(r'\s*–\s*', ', ', text)
    # Clean up any double commas that result (e.g. ", ,")
    text = re.sub(r',\s*,', ',', text)
    return text


def _parse_save_tag(response_text: str) -> tuple[str, dict | None]:
    """
    Look for a <SAVE_ITEM>...</SAVE_ITEM> block in the LLM response.

    Returns
    -------
    (cleaned_text, saved_item | None)
      cleaned_text  : response with the XML block stripped out
      saved_item    : the dict returned by db.save_news_item, or None if no tag found
    """
    pattern = re.compile(r"<SAVE_ITEM>(.*?)</SAVE_ITEM>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(response_text)
    if not match:
        return response_text, None

    block = match.group(1)

    def extract(tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    name = extract("name")
    url = extract("url")
    reason = extract("reason")
    agreed_text = extract("agreed_text")

    if not all([name, url, reason, agreed_text]):
        # Incomplete tag — don't save, just strip the block
        cleaned = pattern.sub("", response_text).strip()
        return cleaned, None

    saved = save_news_item(
        submitter_name=name,
        url=url,
        reason=reason,
        agreed_text=agreed_text,
    )

    cleaned = pattern.sub("", response_text).strip()
    return cleaned, saved


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ManualItemRequest(BaseModel):
    name: str
    url: str
    headline: str
    entry: str
    reason: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "title": settings.app_title},
    )


@app.get("/feed", response_class=HTMLResponse)
def feed_page(request: Request) -> HTMLResponse:
    items = get_feed(limit=200)
    pending = sum(1 for i in items if not i["done"])
    return templates.TemplateResponse(
        "feed.html",
        {
            "request": request,
            "title": settings.app_title,
            "items": items,
            "count": len(items),
            "pending": pending,
        },
    )


@app.get("/api/new-session")
def new_session() -> JSONResponse:
    """
    Create a fresh session and return the standard opening greeting.
    The greeting is hardcoded so it's always exactly right and costs no LLM call.
    """
    session_id = str(uuid.uuid4())
    greeting = (
        "Welcome to the newsletter agent! I'm here to help you add your news item. "
        "If you have no idea how this works, let me know and I'll explain. "
        "Or, if you do know how it works, just tell me your name and we'll get going."
    )
    sessions[session_id] = [{"role": "assistant", "content": greeting}]
    return JSONResponse({"session_id": session_id, "greeting": greeting})


@app.post("/api/chat")
def chat_endpoint(request: Request, body: ChatRequest) -> JSONResponse:
    """
    Process a user message and return the assistant's response.

    Response JSON
    -------------
    {
      "response": str,          # cleaned assistant response (XML tags stripped)
      "saved_item": dict | null # populated if an item was saved this turn
    }
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    history = _get_or_create_session(body.session_id)
    history = _trim_history(history)

    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # If the user is asking about the feed, inject current feed data as context.
    # If they want to update something, try to identify their name from history
    # and inject their recent items so the LLM can help them pick one.
    effective_message = user_message
    if _wants_feed(user_message):
        feed_context = _build_feed_context()
        effective_message = (
            f"{user_message}\n\n"
            f"[SYSTEM NOTE — for assistant only, do not quote verbatim]\n"
            f"{feed_context}"
        )
    # Update/edit flow is currently disabled.
    # elif _wants_update(user_message):
    #     contributor_name = _extract_name_from_history(history)
    #     if contributor_name:
    #         items_context = _build_items_context(contributor_name)
    #         effective_message = (
    #             f"{user_message}\n\n"
    #             f"[SYSTEM NOTE — for assistant only, do not quote verbatim]\n"
    #             f"{items_context}"
    #         )

    try:
        raw_response, updated_history = llm_chat(history, effective_message)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {exc}",
        )

    # Strip em/en dashes before anything else touches the response
    raw_response = _strip_dashes(raw_response)

    # Parse and strip any <SAVE_ITEM> or <UPDATE_ITEM> tag, writing to DB if present
    cleaned_response, saved_item = _parse_save_tag(raw_response)
    if saved_item is None:
        cleaned_response, saved_item = _parse_update_tag(cleaned_response)

    # Persist updated history (use cleaned response so history stays tidy)
    updated_history[-1]["content"] = cleaned_response
    sessions[body.session_id] = updated_history

    return JSONResponse(
        {
            "response": cleaned_response,
            "saved_item": saved_item,
        }
    )


@app.get("/api/feed")
def api_feed() -> JSONResponse:
    """Return the full feed as JSON (for the feed page and any external consumers)."""
    items = get_feed(limit=200)
    return JSONResponse({"count": len(items), "items": items})


@app.post("/api/done/{item_id}")
def set_done(item_id: int, done: bool = True) -> JSONResponse:
    """Toggle the done flag on a news item. Pass ?done=false to undo."""
    mark_done(item_id, done)
    return JSONResponse({"ok": True, "id": item_id, "done": done})


@app.get("/newsletter", response_class=HTMLResponse)
def newsletter_page(request: Request) -> HTMLResponse:
    """Formatted newsletter view — pending (not-done) items only."""
    items = get_feed(limit=200, include_done=False)
    return templates.TemplateResponse(
        "newsletter.html",
        {"request": request, "title": settings.app_title, "items": items},
    )


@app.get("/add-manual", response_class=HTMLResponse)
def add_manual_page(request: Request) -> HTMLResponse:
    """Manual item submission form."""
    return templates.TemplateResponse(
        "add_manual.html",
        {"request": request, "title": settings.app_title},
    )


@app.post("/api/add-manual")
def add_manual(body: ManualItemRequest) -> JSONResponse:
    """
    Save a manually submitted news item.

    Constructs agreed_text as: **{headline}** {entry}
    so the bold-first-sentence convention is preserved across both submission routes.
    """
    # Validate — all fields required
    errors = {}
    if not body.name.strip():
        errors["name"] = "Please enter your name."
    if not body.url.strip():
        errors["url"] = "Please enter a URL."
    if not body.headline.strip():
        errors["headline"] = "Please enter a headline."
    if not body.entry.strip():
        errors["entry"] = "Please enter the entry text."
    if not body.reason.strip():
        errors["reason"] = "Please enter a reason for adding this item."
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    agreed_text = f"**{body.headline.strip()}** {body.entry.strip()}"

    saved = save_news_item(
        submitter_name=body.name.strip(),
        url=body.url.strip(),
        reason=body.reason.strip(),
        agreed_text=agreed_text,
    )
    return JSONResponse({"ok": True, "saved_item": saved})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
