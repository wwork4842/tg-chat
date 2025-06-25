import os
import json
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from telethon import TelegramClient
from telethon.tl.types import User
from telethon.events import NewMessage
from telethon.errors import PeerIdInvalidError
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict
import logging
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
PHONE_NUMBER = os.getenv("TG_PHONE")

client_gpt = AsyncOpenAI(api_key=OPENAI_API_KEY)
client = TelegramClient("web_session", API_ID, API_HASH)
app = FastAPI()
templates = Jinja2Templates(directory="templates")
started = False
auto_reply_users = set()
AUTO_REPLY_FILE = "auto_reply_users.json"
CHAT_HISTORY: Dict[int, List[Dict]] = {}


# Initialize database
def init_db():
    try:
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                     (
                         user_id
                         INTEGER
                         PRIMARY
                         KEY,
                         history
                         TEXT
                     )''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


# Load auto-reply users and chat history from SQLite
def load_data():
    global auto_reply_users, CHAT_HISTORY
    try:
        if os.path.exists(AUTO_REPLY_FILE):
            with open(AUTO_REPLY_FILE, "r") as f:
                auto_reply_users.update(set(json.load(f)))
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute("SELECT user_id, history FROM chat_history")
        CHAT_HISTORY.update({row[0]: json.loads(row[1]) for row in c.fetchall() if row[1] is not None})
        conn.close()
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chat history: {e}. Initializing empty history.")
        CHAT_HISTORY.clear()
    except Exception as e:
        logger.error(f"Error loading data: {e}. Initializing empty history.")
        CHAT_HISTORY.clear()


# Save auto-reply users and chat history to SQLite
def save_data():
    try:
        with open(AUTO_REPLY_FILE, "w") as f:
            json.dump(list(auto_reply_users), f)
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        for user_id, history in CHAT_HISTORY.items():
            c.execute("INSERT OR REPLACE INTO chat_history (user_id, history) VALUES (?, ?)",
                      (user_id, json.dumps(history)))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving data: {e}")


# ChatGPT Response Helper with Context
async def ask_chatgpt(user_id: int, prompt: str, send_message: bool = True) -> str:
    try:
        # Initialize history for new user
        if user_id not in CHAT_HISTORY:
            CHAT_HISTORY[user_id] = [{"role": "system", "content": "You are a helpful assistant."}]

        # Append the new user message
        CHAT_HISTORY[user_id].append({"role": "user", "content": prompt})

        # Send the full conversation history to OpenAI
        response = await client_gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=CHAT_HISTORY[user_id]
        )

        # Append the assistant's response to history
        CHAT_HISTORY[user_id].append({"role": "assistant", "content": response.choices[0].message.content})

        # Save updated history
        save_data()

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error from OpenAI: {e}")
        return "Sorry, I couldn't generate a reply right now."


# Incoming message handler — auto-reply to selected users only
@client.on(NewMessage(incoming=True))
async def handle_message(event):
    sender = await event.get_sender()
    text = event.raw_text
    if sender is None:
        logger.warning("Received message with no sender, skipping auto-reply.")
        return
    if not text.strip() or sender.id not in auto_reply_users:
        return
    response = await ask_chatgpt(sender.id, text)
    await client.send_message(sender.id, response)


# Helper: Fetch last 10 messages for a user
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
                    "timestamp": msg.date.strftime("%Y-%m-%d %H:M:S")
                })
        return messages[::-1]  # Reverse to show newest last
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        return []


# Startup
@app.on_event("startup")
async def startup_event():
    global started
    if not started:
        init_db()  # Initialize database
        load_data()
        await client.connect()
        if not await client.is_user_authorized():
            await client.send_code_request(PHONE_NUMBER)
            code = input("Enter Telegram login code: ")
            await client.sign_in(PHONE_NUMBER, code)
        started = True


# Shutdown
@app.on_event("shutdown")
async def shutdown_event():
    save_data()


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
    return users


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
        custom_user_id: str = Form(""),
        instruction: str = Form(...)
):
    try:
        # Parse custom user IDs (comma-separated)
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                raise ValueError("Custom user IDs must be numeric and comma-separated.")

        # Combine user IDs from checkboxes and custom input
        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")

        # Validate custom IDs
        for target_id in custom_ids:
            try:
                entity = await client.get_entity(target_id)
                if not isinstance(entity, User):
                    raise ValueError(f"The ID {target_id} does not correspond to a user.")
            except PeerIdInvalidError:
                raise ValueError(f"Invalid Telegram user ID {target_id} or the user is not accessible.")

        # Generate and send message to all target users with individual context
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
        custom_user_id: str = Form("")
):
    try:
        # Parse custom user IDs
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                raise ValueError("Custom user IDs must be numeric and comma-separated.")

        # Combine user IDs
        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")

        # Fetch messages for the first target ID
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
        custom_user_id: str = Form("")
):
    try:
        # Parse custom user IDs
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                raise ValueError("Custom user IDs must be numeric and comma-separated.")

        # Combine user IDs
        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")

        # Validate custom IDs
        for target_id in custom_ids:
            try:
                entity = await client.get_entity(target_id)
                if not isinstance(entity, User):
                    raise ValueError(f"The ID {target_id} does not correspond to a user.")
            except PeerIdInvalidError:
                raise ValueError(f"Invalid Telegram user ID {target_id} or the user is not accessible.")

        # Determine toggle action: enable if any are disabled, else disable
        should_enable = any(target_id not in auto_reply_users for target_id in target_ids)
        for target_id in target_ids:
            if should_enable:
                auto_reply_users.add(target_id)
            else:
                auto_reply_users.discard(target_id)
        save_data()

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
        custom_user_id: str = Form(""),
        instruction: str = Form(...)
):
    try:
        # Parse custom user IDs (comma-separated)
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                raise ValueError("Custom user IDs must be numeric and comma-separated.")

        # Combine user IDs from checkboxes and custom input
        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")
        if not instruction.strip():
            raise ValueError("Please provide an instruction for ChatGPT.")

        # Generate preview message for each target_id and use the first for display
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
        custom_user_id: str = Form(""),
        instruction: str = Form(...)
):
    try:
        # Parse custom user IDs (comma-separated)
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                raise ValueError("Custom user IDs must be numeric and comma-separated.")

        # Combine user IDs from checkboxes and custom input
        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")
        if not instruction.strip():
            raise ValueError("Please provide an instruction for context update.")

        # Update context for each target_id and use the first for display
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
        custom_user_id: str = Form(""),
        preview_text: str = Form(...)
):
    logger.info(
        f"Sending preview - user_ids: {user_ids}, custom_user_id: {custom_user_id}, preview_text: {preview_text}")
    try:
        # Parse custom user IDs (comma-separated)
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                raise ValueError("Custom user IDs must be numeric and comma-separated.")

        # Combine user IDs from checkboxes and custom input
        target_ids = list(set(user_ids + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")
        if not preview_text.strip():
            raise ValueError("No preview text available to send.")

        # Send the preview text to all target users
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