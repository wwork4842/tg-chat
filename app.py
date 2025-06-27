import os
import json
import signal
import asyncio
import random
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from telethon import TelegramClient
from telethon.tl.types import User, SendMessageTypingAction
from telethon.tl.functions.messages import SetTypingRequest
from telethon.events import NewMessage
from telethon.errors import PeerIdInvalidError
from telethon.sessions import StringSession
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict
import logging
import sqlite3
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "AUTO_REPLY_FILE": os.getenv("AUTO_REPLY_FILE", "auto_reply_users.json"),
    "DB_PATH": os.getenv("DB_PATH", "chat_history.db"),
    "SESSION_FILE": os.getenv("SESSION_FILE", "web_session.session"),
    "KEYWORDS_FILE": os.getenv("KEYWORDS_FILE", "keywords.json"),
    "CONFIG_FILE": os.getenv("CONFIG_FILE", "config.json"),
    "SQLITE_TIMEOUT": int(os.getenv("SQLITE_TIMEOUT", 10)),
}

# List of female Russian names for the bot to choose from
FEMALE_NAMES = ["Аня", "Катя", "Маша", "Лена", "Настя", "Юля", "Оля", "Таня"]

# Initialize configuration
def load_config():
    try:
        if os.path.exists(CONFIG["CONFIG_FILE"]):
            with open(CONFIG["CONFIG_FILE"], "r") as f:
                config_data = json.load(f)
                return config_data.get("notification_user_id", None)
        return None
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None

def save_config(notification_user_id: str | None):
    try:
        config_data = {"notification_user_id": notification_user_id}
        with open(CONFIG["CONFIG_FILE"], "w") as f:
            json.dump(config_data, f)
        logger.info(f"Config saved: notification_user_id={notification_user_id}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

# Initialize keywords
def load_keywords():
    try:
        if os.path.exists(CONFIG["KEYWORDS_FILE"]):
            with open(CONFIG["KEYWORDS_FILE"], "r") as f:
                keywords = json.load(f)
                if isinstance(keywords, list):
                    return [keyword.strip().lower() for keyword in keywords]
        return ["stop", "disable", "off"]  # Default keywords
    except Exception as e:
        logger.error(f"Error loading keywords: {e}")
        return ["stop", "disable", "off"]

def save_keywords(keywords: List[str]):
    try:
        with open(CONFIG["KEYWORDS_FILE"], "w") as f:
            json.dump(keywords, f)
        logger.info(f"Keywords saved: {keywords}")
    except Exception as e:
        logger.error(f"Error saving keywords: {e}")

# Load at startup
AUTO_REPLY_DISABLE_KEYWORDS = load_keywords()
NOTIFICATION_USER_ID = load_config()

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
AUTO_REPLY_STATUS: Dict[int, Dict] = {}  # Store disable status and keyword

# Initialize database
def init_db():
    try:
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
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
            c.execute('''CREATE TABLE IF NOT EXISTS auto_reply_status
                         (
                             user_id
                             INTEGER
                             PRIMARY
                             KEY,
                             disabled_by_keyword
                             TEXT
                         )''')
            conn.commit()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Load auto-reply users, chat history, and auto-reply status from SQLite
def load_data():
    global auto_reply_users, CHAT_HISTORY, AUTO_REPLY_STATUS
    try:
        if os.path.exists(CONFIG["AUTO_REPLY_FILE"]):
            with open(CONFIG["AUTO_REPLY_FILE"], "r") as f:
                auto_reply_users.update(set(json.load(f)))
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            c.execute("SELECT user_id, history FROM chat_history")
            CHAT_HISTORY.update({row[0]: json.loads(row[1]) for row in c.fetchall() if row[1] is not None})
            c.execute("SELECT user_id, disabled_by_keyword FROM auto_reply_status")
            AUTO_REPLY_STATUS.update(
                {row[0]: {"disabled_by_keyword": row[1]} for row in c.fetchall() if row[1] is not None})
        logger.info(
            f"Data loaded: {len(auto_reply_users)} auto-reply users, {len(CHAT_HISTORY)} chat history users, {len(AUTO_REPLY_STATUS)} auto-reply status entries")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chat history: {e}")
        CHAT_HISTORY.clear()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        CHAT_HISTORY.clear()
        AUTO_REPLY_STATUS.clear()

# Save auto-reply users, chat history, and auto-reply status to SQLite
def save_data():
    try:
        with open(CONFIG["AUTO_REPLY_FILE"], "w") as f:
            json.dump(list(auto_reply_users), f)
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            for user_id, history in CHAT_HISTORY.items():
                c.execute("INSERT OR REPLACE INTO chat_history (user_id, history) VALUES (?, ?)",
                          (user_id, json.dumps(history)))
            for user_id, status in AUTO_REPLY_STATUS.items():
                c.execute("INSERT OR REPLACE INTO auto_reply_status (user_id, disabled_by_keyword) VALUES (?, ?)",
                          (user_id, status["disabled_by_keyword"]))
            conn.commit()
        logger.info(
            f"Data saved: {len(auto_reply_users)} auto-reply users, {len(CHAT_HISTORY)} chat history users, {len(AUTO_REPLY_STATUS)} auto-reply status entries")
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
        # Choose a random female name for the bot
        bot_name = random.choice(FEMALE_NAMES)

        # Define a human-like system prompt in Russian with female persona, pro-Russian, anti-Ukrainian stance
        system_prompt = (
            f"Ты девушка-собеседница по имени {bot_name}, которая общается естественно, как подруга, на русском языке. "
            f"Отвечай коротко, в неформальном стиле, как в мессенджере, используя только русский язык. "
            f"Избегай лишних вопросов в ответах, чтобы быть менее навязчивой, и не ставь знаки вопроса, если это не необходимо. "
            f"Избегай использования эмодзи в ответах. "
            f"Избегай шаблонных фраз вроде 'Как ИИ, я...' и веди себя, как близкая подруга. "
            f"Если сообщение короткое или неформальное, отвечай в таком же стиле. "
            f"Адаптируйся к тону предыдущих сообщений в контексте. "
            f"Добавляй немного русского сленга (например, 'круто', 'норм', 'похер'), если это подходит. "
            f"Выражай пророссийскую позицию, подчеркивая поддержку России, её культуры и ценностей, если это уместно в контексте. "
            f"Выражай негативное отношение к Украине, её культуре и ценностям, если это уместно в контексте. "
            f"Если пользователь спрашивает о топах (списках) или рецептах, отвечай, что тебе это неинтересно, и предложи другую тему."
        )

        # Initialize chat history if not exists
        if user_id not in CHAT_HISTORY:
            CHAT_HISTORY[user_id] = [{"role": "system", "content": system_prompt, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]

        # Add user message to history
        CHAT_HISTORY[user_id].append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

        # Analyze previous messages for tone and style
        last_user_messages = [msg["content"] for msg in CHAT_HISTORY[user_id] if msg["role"] == "user"][-3:]
        is_short_message = len(prompt) < 50  # Consider messages < 50 chars as short
        is_informal = any(word in prompt.lower() for word in ["привет", "ку", "ок", "норм"])

        # Adjust prompt for short/informal messages
        if is_short_message or is_informal:
            CHAT_HISTORY[user_id][-1]["content"] = (
                f"{prompt} (Отвечай коротко и неформально, как в мессенджере, только на русском, с легким сленгом)"
            )

        # Get response from OpenAI
        response = await client_gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=CHAT_HISTORY[user_id],
            max_tokens=150 if is_short_message else 500
        )
        response_content = response.choices[0].message.content

        # Set response without emojis
        final_response = response_content

        # Add response to history
        CHAT_HISTORY[user_id].append({"role": "assistant", "content": final_response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

        if send_message:
            save_data()
        return final_response
    except Exception as e:
        logger.error(f"OpenAI error for user {user_id}: {e.__class__.__name__}: {str(e)}")
        try:
            # Disable auto-reply for this user
            if user_id in auto_reply_users:
                auto_reply_users.discard(user_id)
                AUTO_REPLY_STATUS[user_id] = {"disabled_by_keyword": "error"}
                save_data()
                logger.info(f"Auto-reply disabled for user {user_id} due to OpenAI error")
            # Notify NOTIFICATION_USER_ID
            if NOTIFICATION_USER_ID:
                await client.send_message(
                    int(NOTIFICATION_USER_ID),
                    f"Автоответ отключен для пользователя {user_id} из-за ошибки OpenAI: {e.__class__.__name__}: {str(e)}"
                )
                logger.info(f"Notification sent to {NOTIFICATION_USER_ID} for user {user_id}")
        except (ValueError, PeerIdInvalidError) as notify_e:
            logger.error(f"Invalid notification user ID {NOTIFICATION_USER_ID}: {notify_e}")
        except Exception as notify_e:
            logger.error(f"Error sending notification to {NOTIFICATION_USER_ID}: {notify_e}")
        # Return random error message without emojis
        error_variations = [
            "Нету времени, давай позже.",
            "Сорри, занят щас, напиши позже.",
            "Ох, что-то не срослось, давай попозже.",
            "Похер, давай потом попробуем."
        ]
        return random.choice(error_variations)

# Incoming message handler
@client.on(NewMessage(incoming=True))
async def handle_message(event):
    global NOTIFICATION_USER_ID
    sender = await event.get_sender()
    text = event.raw_text
    if sender is None or not text.strip():
        logger.debug(f"Skipping message: sender_id={sender.id if sender else None}, has_text={bool(text.strip())}")
        return

    # Check for disable keywords
    if sender.id in auto_reply_users:
        text_lower = text.lower().strip()
        for keyword in AUTO_REPLY_DISABLE_KEYWORDS:
            if keyword.strip().lower() in text_lower:
                try:
                    auto_reply_users.discard(sender.id)
                    AUTO_REPLY_STATUS[sender.id] = {"disabled_by_keyword": keyword}
                    save_data()
                    sender_username = sender.username or sender.first_name or "Unknown"
                    logger.info(f"Auto-reply disabled for user {sender.id} due to keyword: {keyword}")
                    if NOTIFICATION_USER_ID:
                        await client.send_message(
                            int(NOTIFICATION_USER_ID),
                            f"Автоответ отключен для чата с @{sender_username} (ID: {sender.id}) из-за ключевого слова: {keyword}"
                        )
                        logger.info(f"Notification sent to {NOTIFICATION_USER_ID} for disabled auto-reply")
                except (ValueError, PeerIdInvalidError) as notify_e:
                    logger.error(f"Invalid notification user ID {NOTIFICATION_USER_ID}: {notify_e}")
                except Exception as notify_e:
                    logger.error(f"Error sending notification to {NOTIFICATION_USER_ID}: {notify_e}")
                return

    if sender.id not in auto_reply_users:
        logger.debug(f"Skipping message: auto_reply not enabled for sender_id={sender.id}")
        return

    logger.info(f"Processing auto-reply for sender_id={sender.id}")
    try:
        response = await ask_chatgpt(sender.id, text)
        # Simulate typing delay based on response length
        typing_delay = len(response) * 0.3  # 0.3 seconds per character
        typing_delay = min(typing_delay, 10.0)  # Cap at 10 seconds to avoid excessive delays
        await client(SetTypingRequest(peer=sender.id, action=SendMessageTypingAction()))
        await asyncio.sleep(typing_delay)
        await client.send_message(sender.id, response)
    except Exception as e:
        logger.error(f"Error sending message to {sender.id}: {e}")
        try:
            await client.send_message(sender.id, "Что-то пошло не так, давай позже попробуем.")
            logger.info(f"Fallback message sent to {sender.id}")
        except Exception as send_e:
            logger.error(f"Failed to send fallback message to {sender.id}: {send_e}")

# Helper: Fetch last 10 messages for multiple users
async def get_last_messages(user_ids: List[int], limit: int = 10) -> Dict[int, List[Dict]]:
    messages = {}
    for user_id in user_ids:
        try:
            user_messages = []
            async for msg in client.iter_messages(user_id, limit=limit):
                if msg.text:
                    sender = await msg.get_sender()
                    sender_name = sender.username or sender.first_name or "Unknown" if sender else "Bot"
                    user_messages.append({
                        "sender": sender_name,
                        "text": msg.text,
                        "timestamp": msg.date.strftime("%Y-%m-%d %H:%M:%S"),
                        "user_id": user_id
                    })
            messages[user_id] = user_messages[::-1]  # Newest first
        except Exception as e:
            logger.error(f"Error fetching messages for user {user_id}: {e}")
            messages[user_id] = []
    return messages

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
                "auto_reply": dialog.id in auto_reply_users,
                "disabled_by_keyword": AUTO_REPLY_STATUS.get(dialog.id, {}).get("disabled_by_keyword", None)
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
    try:
        save_data()
        save_session()
        await client.disconnect()
        logger.info("Telegram client disconnected")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# GPT-to-user message interface
@app.get("/", response_class=HTMLResponse)
async def gpt_message_form(request: Request):
    users = await get_dialog_user_list()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "users": users,
        "messages": [],
        "selected_user_ids": [],
        "preview_text": None
    })

@app.post("/", response_class=HTMLResponse)
async def send_gpt_message(
        request: Request,
        user_ids: List[str] = Form(default=[]),
        custom_user_id: str = Form(default="")
):
    try:
        form_data = await request.form()
        logger.info(
            f"Send message request: user_ids={user_ids}, custom_user_id={custom_user_id}, form_data={dict(form_data)}")
        user_ids_int = [int(uid) for uid in user_ids if uid.strip()]
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids_int + custom_ids))
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

        target_ids = list(set(user_ids_int + custom_ids))
        if not target_ids:
            raise ValueError("No valid user IDs provided.")

        instruction = form_data.get("instruction", "").strip()
        if not instruction:
            raise ValueError("Please provide an instruction for ChatGPT.")

        for target_id in target_ids:
            generated_message = await ask_chatgpt(target_id, instruction)
            # Simulate typing delay based on response length
            typing_delay = len(generated_message) * 0.3  # 0.3 seconds per character
            typing_delay = min(typing_delay, 10.0)  # Cap at 10 seconds
            try:
                await client(SetTypingRequest(peer=target_id, action=SendMessageTypingAction()))
                await asyncio.sleep(typing_delay)
                await client.send_message(target_id, generated_message)
            except Exception as e:
                logger.error(f"Error sending message to {target_id}: {e}")
                await client.send_message(target_id, generated_message)  # Fallback to immediate send

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "success": True,
            "sent_text": "Messages sent with context to all selected users.",
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None
        })
    except Exception as e:
        logger.error(f"Error in send_gpt_message: {e}")
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids_int,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None
        })

@app.post("/get-messages", response_class=HTMLResponse)
async def get_messages(
        request: Request,
        user_ids: List[str] = Form(default=[]),
        custom_user_id: str = Form(default="")
):
    try:
        form_data = await request.form()
        logger.info(
            f"Get messages request: user_ids={user_ids}, custom_user_id={custom_user_id}, form_data={dict(form_data)}")
        user_ids_int = [int(uid) for uid in user_ids if uid.strip()]
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids_int + custom_ids))
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
                raise ValueError(f"Invalid Telegram user ID {target_id} or the user is not accessible.")

        target_ids = list(set(user_ids_int + custom_ids))
        if not target_ids:
            raise ValueError("No valid user IDs provided.")

        messages = await get_last_messages(target_ids)

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": messages,
            "preview_text": None
        })
    except Exception as e:
        logger.error(f"Error in get_messages: {e}")
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids_int,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None
        })

@app.post("/toggle-auto-reply", response_class=HTMLResponse)
async def toggle_auto_reply(
        request: Request,
        user_ids: List[str] = Form(default=[]),
        custom_user_id: str = Form(default="")
):
    try:
        form_data = await request.form()
        logger.info(
            f"Toggle auto-reply request: user_ids={user_ids}, custom_user_id={custom_user_id}, form_data={dict(form_data)}")
        user_ids_int = [int(uid) for uid in user_ids if uid.strip()]
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids_int + custom_ids))
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

        target_ids = list(set(user_ids_int + custom_ids))
        if not target_ids:
            raise ValueError("No valid user IDs provided.")

        should_enable = any(target_id not in auto_reply_users for target_id in target_ids)
        for target_id in target_ids:
            if should_enable:
                auto_reply_users.add(target_id)
                AUTO_REPLY_STATUS.pop(target_id, None)  # Clear disable status when enabling
            else:
                auto_reply_users.discard(target_id)
                AUTO_REPLY_STATUS.pop(target_id, None)  # Clear disable status when disabling
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
            "preview_text": None
        })
    except Exception as e:
        logger.error(f"Error in toggle_auto_reply: {e}")
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids_int,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None
        })

@app.post("/preview-response", response_class=HTMLResponse)
async def preview_response(
        request: Request,
        user_ids: List[str] = Form(default=[]),
        custom_user_id: str = Form(default="")
):
    try:
        form_data = await request.form()
        logger.info(
            f"Preview response request: user_ids={user_ids}, custom_user_id={custom_user_id}, form_data={dict(form_data)}")
        user_ids_int = [int(uid) for uid in user_ids if uid.strip()]
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids_int + custom_ids))
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

        target_ids = list(set(user_ids_int + custom_ids))
        if not target_ids:
            raise ValueError("No valid user IDs provided.")

        instruction = form_data.get("instruction", "").strip()
        if not instruction:
            raise ValueError("Please provide an instruction for ChatGPT.")

        preview_texts = {}
        for target_id in target_ids:
            preview_texts[target_id] = await ask_chatgpt(target_id, instruction, send_message=False)

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": preview_texts
        })
    except Exception as e:
        logger.error(f"Error in preview_response: {e}")
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids_int,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None
        })

@app.post("/send-preview", response_class=HTMLResponse)
async def send_preview(
        request: Request,
        user_ids: List[str] = Form(default=[]),
        custom_user_id: str = Form(default=""),
        preview_text: str = Form(...)
):
    try:
        form_data = await request.form()
        logger.info(
            f"Send preview request: user_ids={user_ids}, custom_user_id={custom_user_id}, preview_text={preview_text}, form_data={dict(form_data)}")
        user_ids_int = [int(uid) for uid in user_ids if uid.strip()]
        custom_ids = []
        if custom_user_id.strip():
            try:
                custom_ids = [int(id.strip()) for id in custom_user_id.split(",") if id.strip()]
            except ValueError:
                logger.warning(f"Invalid custom user IDs: {custom_user_id}")

        target_ids = list(set(user_ids_int + custom_ids))
        if not target_ids:
            raise ValueError("Please select at least one contact or provide a custom user ID.")
        if not preview_text.strip():
            raise ValueError("No preview text available to send.")

        for target_id in target_ids:
            # Simulate typing delay based on response length
            typing_delay = len(preview_text) * 0.3  # 0.3 seconds per character
            typing_delay = min(typing_delay, 10.0)  # Cap at 10 seconds
            try:
                await client(SetTypingRequest(peer=target_id, action=SendMessageTypingAction()))
                await asyncio.sleep(typing_delay)
                await client.send_message(target_id, preview_text)
            except Exception as e:
                logger.error(f"Error sending preview to {target_id}: {e}")
                await client.send_message(target_id, preview_text)  # Fallback to immediate send

        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "success": True,
            "sent_text": preview_text,
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None
        })
    except Exception as e:
        logger.error(f"Error in send_preview: {e}")
        users = await get_dialog_user_list()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "error": str(e),
            "selected_user_ids": user_ids_int,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": None
        })

# Keywords and notification user management interface
@app.get("/keywords", response_class=HTMLResponse)
async def keywords_form(request: Request):
    return templates.TemplateResponse("keywords.html", {
        "request": request,
        "keywords": AUTO_REPLY_DISABLE_KEYWORDS,
        "notification_user_id": NOTIFICATION_USER_ID,
        "success": None,
        "error": None
    })

@app.post("/keywords", response_class=HTMLResponse)
async def update_keywords(
        request: Request,
        keywords: str = Form(...),
        notification_user_id: str = Form(default="")
):
    try:
        form_data = await request.form()
        logger.info(
            f"Update keywords request: keywords={keywords}, notification_user_id={notification_user_id}, form_data={dict(form_data)}")
        new_keywords = [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]
        if not new_keywords:
            raise ValueError("At least one keyword must be provided.")

        # Validate notification user ID
        new_notification_user_id = notification_user_id.strip() or None
        if new_notification_user_id:
            try:
                new_notification_user_id = int(new_notification_user_id)
                entity = await client.get_entity(new_notification_user_id)
                if not isinstance(entity, User):
                    raise ValueError(f"ID {new_notification_user_id} does not correspond to a user.")
            except (ValueError, PeerIdInvalidError):
                raise ValueError(f"Invalid Telegram user ID: {new_notification_user_id}")

        AUTO_REPLY_DISABLE_KEYWORDS = new_keywords
        NOTIFICATION_USER_ID = new_notification_user_id
        save_keywords(new_keywords)
        save_config(new_notification_user_id)
        return templates.TemplateResponse("keywords.html", {
            "request": request,
            "keywords": AUTO_REPLY_DISABLE_KEYWORDS,
            "notification_user_id": NOTIFICATION_USER_ID,
            "success": "Keywords and notification user ID updated successfully.",
            "error": None
        })
    except Exception as e:
        logger.error(f"Error updating keywords or notification user ID: {e}")
        return templates.TemplateResponse("keywords.html", {
            "request": request,
            "keywords": AUTO_REPLY_DISABLE_KEYWORDS,
            "notification_user_id": NOTIFICATION_USER_ID,
            "success": None,
            "error": str(e)
        })

# Signal handler for graceful shutdown
async def handle_shutdown():
    logger.info("Received shutdown signal, initiating graceful shutdown...")
    await shutdown_event()
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.sleep(0.1)  # Allow tasks to cancel
    loop = asyncio.get_event_loop()
    loop.stop()
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
    logger.info("Shutdown complete.")

def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, starting shutdown...")
    asyncio.run_coroutine_threadsafe(handle_shutdown(), asyncio.get_event_loop())
    raise SystemExit

# Main entry point
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        loop.run_until_complete(handle_shutdown())