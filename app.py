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
from typing import List

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

# Load auto-reply users from JSON file
def load_auto_reply_users():
    global auto_reply_users
    try:
        if os.path.exists(AUTO_REPLY_FILE):
            with open(AUTO_REPLY_FILE, "r") as f:
                auto_reply_users.update(set(json.load(f)))
    except Exception as e:
        print(f"Error loading auto-reply users: {e}")

# Save auto-reply users to JSON file
def save_auto_reply_users():
    try:
        with open(AUTO_REPLY_FILE, "w") as f:
            json.dump(list(auto_reply_users), f)
    except Exception as e:
        print(f"Error saving auto-reply users: {e}")

# ChatGPT Response Helper
async def ask_chatgpt(prompt: str) -> str:
    try:
        response = await client_gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error from OpenAI: {e}")
        return "Sorry, I couldn't generate a reply right now."

# Incoming message handler — auto-reply to selected users only
@client.on(NewMessage(incoming=True))
async def handle_message(event):
    sender = await event.get_sender()
    text = event.raw_text
    if not text.strip() or sender.id not in auto_reply_users:
        return
    response = await ask_chatgpt(text)
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
                    "timestamp": msg.date.strftime("%Y-%m-%d %H:%M:%S")
                })
        return messages[::-1]  # Reverse to show newest last
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []

# Startup
@app.on_event("startup")
async def startup_event():
    global started
    if not started:
        load_auto_reply_users()
        await client.connect()
        if not await client.is_user_authorized():
            await client.send_code_request(PHONE_NUMBER)
            code = input("Enter Telegram login code: ")
            await client.sign_in(PHONE_NUMBER, code)
        started = True

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
        "selected_user_ids": []
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

        # Generate and send message to all target users
        generated_message = await ask_chatgpt(
            f"Write a polite and clear message to the user based on this instruction: {instruction}"
        )
        for target_id in target_ids:
            await client.send_message(target_id, generated_message)

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "success": True,
            "sent_text": generated_message,
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": []
        })
    except Exception as e:
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": []
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
            "messages": messages
        })
    except Exception as e:
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": []
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
        save_auto_reply_users()

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "success": True,
            "sent_text": f"Auto-reply {'enabled' if should_enable else 'disabled'} for selected users.",
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": []
        })
    except Exception as e:
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids,
            "custom_user_id": custom_user_id,
            "messages": []
        })