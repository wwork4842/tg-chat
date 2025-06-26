import os
import json
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from telethon import TelegramClient
from telethon.tl.types import User
from telethon.events import NewMessage
from telethon.errors import PeerIdInvalidError
from telethon.sessions import StringSession
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict
import logging
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "AUTO_REPLY_FILE": os.getenv("AUTO_REPLY_FILE", "auto_reply_users.json"),
    "DB_PATH": os.getenv("DB_PATH", "chat_history.db"),
    "SESSION_FILE": os.getenv("SESSION_FILE", "web_session.session"),
    "SQLITE_TIMEOUT": int(os.getenv("SQLITE_TIMEOUT", 10)),
}

# Validate environment variables
def validate_env_vars():
    required_vars = ["OPENAI_API_KEY", "TG_API_ID", "TG_API_HASH", "TG_PHONE"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

load_dotenv()
validate_env_vars()

# Load or initialize session
session_string = None
if os.path.exists(CONFIG["SESSION_FILE"]):
    try:
        with open(CONFIG["SESSION_FILE"], "r") as f:
            session_string = f.read().strip()
        logger.info(f"Loaded session from {CONFIG['SESSION_FILE']}")
    except Exception as e:
        logger.error(f"Failed to load session file: {e}")

# Initialize clients
client_gpt = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = TelegramClient(
    StringSession(session_string),
    int(os.getenv("TG_API_ID")),
    os.getenv("TG_API_HASH")
)
app = FastAPI()
templates = Jinja2Templates(directory="templates")
started = False
auto_reply_users = set()
CHAT_HISTORY: Dict[int, List[Dict]] = {}

# Initialize database
def init_db():
    try:
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                         (user_id INTEGER PRIMARY KEY, history TEXT)''')
            conn.commit()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Load auto-reply users and chat history from SQLite
def load_data():
    global auto_reply_users, CHAT_HISTORY
    try:
        if os.path.exists(CONFIG["AUTO_REPLY_FILE"]):
            with open(CONFIG["AUTO_REPLY_FILE"], "r") as f:
                auto_reply_users.update(set(json.load(f)))
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            c.execute("SELECT user_id, history FROM chat_history")
            CHAT_HISTORY.update({row[0]: json.loads(row[1]) for row in c.fetchall() if row[1] is not None})
        logger.info(f"Data loaded: {len(auto_reply_users)} auto-reply users, {len(CHAT_HISTORY)} chat history users")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chat history: {e}")
        CHAT_HISTORY.clear()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        CHAT_HISTORY.clear()

# Save auto-reply users and chat history to SQLite
def save_data():
    try:
        with open(CONFIG["AUTO_REPLY_FILE"], "w") as f:
            json.dump(list(auto_reply_users), f)
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            for user_id, history in CHAT_HISTORY.items():
                c.execute("INSERT OR REPLACE INTO chat_history (user_id, history) VALUES (?, ?)",
                          (user_id, json.dumps(history)))
            conn.commit()
        logger.info(f"Data saved: {len(auto_reply_users)} auto-reply users, {len(CHAT_HISTORY)} chat history users")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

# Save session
def save_session():
    try:
        session_string = client.session.save()
        with open(CONFIG["SESSION_FILE"], "w") as f:
            f.write(session_string)
        logger.info(f"Session saved to {CONFIG['SESSION_FILE']}")
    except Exception as e:
        logger.error(f"Failed to save session: {e}")

# ChatGPT Response Helper with Context
async def ask_chatgpt(user_id: int, prompt: str, send_message: bool = True) -> str:
    try:
        if user_id not in CHAT_HISTORY:
            CHAT_HISTORY[user_id] = [{"role": "system", "content": "You are a helpful assistant."}]
        CHAT_HISTORY[user_id].append({"role": "user", "content": prompt})
        response = await client_gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=CHAT_HISTORY[user_id]
        )
        response_content = response.choices[0].message.content
        CHAT_HISTORY[user_id].append({"role": "assistant", "content": response_content})
        if send_message:
            save_data()
        return response_content
    except Exception as e:
        logger.error(f"OpenAI error for user {user_id}: {e}")
        return "Sorry, I couldn't generate a reply right now."

# Incoming message handler
@client.on(NewMessage(incoming=True))
async def handle_message(event):
    sender = await event.get_sender()
    text = event.raw_text
    if sender is None or not text.strip() or sender.id not in auto_reply_users:
        logger.debug(f"Skipping message: sender_id={sender.id if sender else None}, has_text={bool(text.strip())}, auto_reply_enabled={sender.id in auto_reply_users if sender else False}")
        return
    logger.info(f"Processing auto-reply for sender_id={sender.id}")
    response = await ask_chatgpt(sender.id, text)
    await client.send_message(sender.id, response)

# Helper: Fetch last 10 messages
async def get_last_messages(user_id: int, limit: int = 10):
    try:
        messages = []
        async for msg in client.iter_messages(user_id, limit=limit):
            if msg.text:
                sender = await msg.get_sender()
                sender_name = sender.username or sender.first_name or "Unknown" if sender else "Bot"
                messages.append({
                    "sender": sender_name,
                    "text": msg.text,
                    "timestamp": msg.date.strftime("%Y-%m-%d %H:%M:%S")  # Fixed timestamp format
                })
        return messages[::-1]  # Newest first
    except Exception as e:
        logger.error(f"Error fetching messages for user {user_id}: {e}")
        return []

# Helper: Users only
async def get_dialog_user_list():
    users = []
    async for dialog in client.iter_dialogs():
        entity = dialog.entity
        if isinstance(entity, User) and not entity.bot:
            name = f"{dialog.name} (@{entity.username})" if entity.username else dialog.name
            users.append({
                "id": dialog.id,
                "name": name,
                "auto_reply": dialog.id in auto_reply_users
            })
    logger.info(f"Dialog list retrieved: {len(users)} users")
    return users

# Startup
@app.on_event("startup")
async def startup_event():
    global started
    if not started:
        init_db()
        load_data()
        try:
            await client.connect()
            if not await client.is_user_authorized():
                await client.send_code_request(os.getenv("TG_PHONE"))
                code = input("Enter Telegram login code: ")
                await client.sign_in(os.getenv("TG_PHONE"), code)
                save_session()
            logger.info("Telegram client connected and authorized")
        except Exception as e:
            logger.error(f"Failed to connect Telegram client: {e}")
            raise
        started = True

# Shutdown
@app.on_event("shutdown")
async def shutdown_event():
    save_data()
    save_session()
    await client.disconnect()
    logger.info("Telegram client disconnected")

# GPT-to-user message interface
@app.get("/", response_class=HTMLResponse)
async def gpt_message_form(request: Request):
    users = await get_dialog_user_list()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "users": users,
        "messages": [],
        "selected_user_ids": [],
        "preview_text": None,
        "context_preview_text": None
    })

@app.post("/", response_class=HTMLResponse)
async def send_gpt_message(
        request: Request,
        user_ids: List[int] = Form(default=[]),
        custom_user_id: str = Form(default=""),
        instruction: str = Form(default="")
):
    try:
        logger.info(f"Send message request: user_ids={user_ids}, custom_user_id={custom_user_id}, instruction={instruction}")
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")

        for target_id in custom_ids:
            try:
                entity = await client.get_entity(target_id)
                if not isinstance(entity, User):
                    logger.warning(f"ID is not a user: {target_id}")
                    if target_id in custom_ids:
                        custom_ids.remove(target_id)
            except PeerIdInvalidError:
                logger.warning(f"Invalid Telegram user ID: {target_id}")
                if target_id in custom_ids:
                    custom_ids.remove(target_id)

        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("No valid user IDs provided.")

        if not instruction.strip():
            raise ValueError("Please provide an instruction for ChatGPT.")

        for target_id in target_ids:
            generated_message = await ask_chatgpt(target_id, instruction)
            await client.send_message(target_id, generated_message)

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "success": True,
            "sent_text": "Messages sent with context to all selected users.",
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })
    except Exception as e:
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })

@app.post("/get-messages", response_class=HTMLResponse)
async def get_messages(
        request: Request,
        user_ids: List[int] = Form(default=[]),
        custom_user_id: str = Form(default="")
):
    try:
        logger.info(f"Get messages request: user_ids={user_ids}, custom_user_id={custom_user_id}")
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")

        target_id = target_ids[0]
        if target_id in custom_ids:
            try:
                entity = await client.get_entity(target_id)
                if not isinstance(entity, User):
                    raise ValueError(f"The ID {target_id} does not correspond to a user.")
            except PeerIdInvalidError:
                raise ValueError(f"Invalid Telegram user ID {target_id} or the user is not accessible.")

        messages = await get_last_messages(target_id)

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": messages,
            "preview_text": None,
            "context_preview_text": None
        })
    except Exception as e:
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })

@app.post("/toggle-auto-reply", response_class=HTMLResponse)
async def toggle_auto_reply(
        request: Request,
        user_ids: List[int] = Form(default=[]),
        custom_user_id: str = Form(default="")
):
    try:
        logger.info(f"Toggle auto-reply request: user_ids={user_ids}, custom_user_id={custom_user_id}")
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")

        for target_id in custom_ids:
            try:
                entity = await client.get_entity(target_id)
                if not isinstance(entity, User):
                    logger.warning(f"ID is not a user: {target_id}")
                    if target_id in custom_ids:
                        custom_ids.remove(target_id)
            except PeerIdInvalidError:
                logger.warning(f"Invalid Telegram user ID: {target_id}")
                if target_id in custom_ids:
                    custom_ids.remove(target_id)

        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("No valid user IDs provided.")

        should_enable = any(target_id not in auto_reply_users for target_id in target_ids)
        for target_id in target_ids:
            if should_enable:
                auto_reply_users.add(target_id)
            else:
                auto_reply_users.discard(target_id)
        save_data()
        logger.info(f"Auto-reply toggled: should_enable={should_enable}, auto_reply_users={list(auto_reply_users)}")

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "success": True,
            "sent_text": f"Auto-reply {'enabled' if should_enable else 'disabled'} for selected users.",
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })
    except Exception as e:
        logger.error(f"Error in toggle_auto_reply: {e}")
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })

@app.post("/preview-response", response_class=HTMLResponse)
async def preview_response(
        request: Request,
        user_ids: List[int] = Form(default=[]),
        custom_user_id: str = Form(default=""),
        instruction: str = Form(default="")
):
    try:
        logger.info(f"Preview response request: user_ids={user_ids}, custom_user_id={custom_user_id}, instruction={instruction}")
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")
        if not instruction.strip():
            raise ValueError("Please provide an instruction for ChatGPT.")

        preview_texts = {}
        for target_id in target_ids:
            preview_texts[target_id] = await ask_chatgpt(target_id, instruction, send_message=False)
        preview_text = preview_texts.get(target_ids[0], "No preview generated.")

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": preview_text,
            "context_preview_text": None
        })
    except Exception as e:
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })

@app.post("/update-context", response_class=HTMLResponse)
async def update_context(
        request: Request,
        user_ids: List[int] = Form(default=[]),
        custom_user_id: str = Form(default=""),
        instruction: str = Form(default="")
):
    try:
        logger.info(f"Update context request: user_ids={user_ids}, custom_user_id={custom_user_id}, instruction={instruction}")
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")
        if not instruction.strip():
            raise ValueError("Please provide an instruction for context update.")

        context_preview_texts = {}
        for target_id in target_ids:
            context_preview_texts[target_id] = await ask_chatgpt(target_id, instruction, send_message=False)
        context_preview_text = context_preview_texts.get(target_ids[0], "No context update generated.")

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": context_preview_text
        })
    except Exception as e:
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })

@app.post("/send-preview", response_class=HTMLResponse)
async def send_preview(
        request: Request,
        user_ids: List[int] = Form(default=[]),
        custom_user_id: str = Form(default=""),
        preview_text: str = Form(...)
):
    try:
        logger.info(f"Send preview request: user_ids={user_ids}, custom_user_id={custom_user_id}, preview_text={preview_text}")
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")
        if not preview_text.strip():
            raise ValueError("No preview text available to send.")

        for target_id in target_ids:
            await client.send_message(target_id, preview_text)

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "success": True,
            "sent_text": preview_text,
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })
    except Exception as e:
        logger.error(f"Error in send_preview: {e}")
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None,
            "context_preview_text": None
        })