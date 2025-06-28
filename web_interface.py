import json
import sqlite3
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Optional, Tuple
from telethon.tl.types import User, SendMessageTypingAction
from telethon.tl.functions.messages import SetTypingRequest
from telethon.errors import PeerIdInvalidError
import asyncio
import os
from utils import logger, get_dialog_user_list, get_last_messages, save_data, save_keywords, save_config, \
    auto_reply_users, AUTO_REPLY_STATUS, NOTIFICATION_USER_ID, AUTO_REPLY_DISABLE_KEYWORDS, reset_chat_history, \
    CHAT_HISTORY, CONFIG, FEMALE_NAMES
from chatgpt_handler import ask_chatgpt
import random
from datetime import datetime
from functools import wraps

# Constants
TYPING_DELAY_PER_CHAR = 0.3
MAX_TYPING_DELAY = 10.0
REQUEST_TIMEOUT = 15.0
ENTITY_TIMEOUT = 5.0

# Initialize templates
templates = Jinja2Templates(directory="templates")


class TelegramService:
    """Service class for Telegram operations"""

    def __init__(self, client):
        self.client = client

    async def send_message_with_typing(self, target_id: int, message: str) -> None:
        """Send message with realistic typing simulation"""
        typing_delay = min(len(message) * TYPING_DELAY_PER_CHAR, MAX_TYPING_DELAY)
        try:
            await self.client(SetTypingRequest(peer=target_id, action=SendMessageTypingAction()))
            await asyncio.sleep(typing_delay)
            await self.client.send_message(target_id, message)
        except Exception as e:
            logger.error(f"Error sending message to {target_id}: {e}")
            # Fallback to immediate send
            await self.client.send_message(target_id, message)

    async def validate_user_entity(self, user_id: int) -> bool:
        """Validate if user ID corresponds to a real Telegram user"""
        try:
            entity = await asyncio.wait_for(self.client.get_entity(user_id), timeout=ENTITY_TIMEOUT)
            return isinstance(entity, User)
        except (asyncio.TimeoutError, PeerIdInvalidError, ValueError):
            return False


class UserService:
    """Service class for user operations"""

    @staticmethod
    async def validate_and_process_user_ids(
            user_ids: List[str],
            custom_user_id: str,
            telegram_service: TelegramService
    ) -> Tuple[List[int], List[int]]:
        """Extract and validate user IDs from form data"""
        # Process selected user IDs
        user_ids_int = []
        for uid in user_ids:
            if uid.strip():
                try:
                    user_ids_int.append(int(uid))
                except ValueError:
                    logger.warning(f"Invalid user ID: {uid}")

        # Process custom user IDs
        custom_ids = []
        if custom_user_id.strip():
            try:
                raw_ids = [id.strip() for id in custom_user_id.split(",") if id.strip()]
                for raw_id in raw_ids:
                    try:
                        custom_ids.append(int(raw_id))
                    except ValueError:
                        logger.warning(f"Invalid custom user ID: {raw_id}")
            except Exception as e:
                logger.warning(f"Error processing custom user IDs: {e}")

        # Validate custom user IDs against Telegram
        validated_custom_ids = []
        for custom_id in custom_ids:
            if await telegram_service.validate_user_entity(custom_id):
                validated_custom_ids.append(custom_id)
            else:
                logger.warning(f"Invalid Telegram user ID: {custom_id}")

        # Combine and deduplicate
        target_ids = list(set(user_ids_int + validated_custom_ids))

        if not target_ids:
            raise ValueError("No valid user IDs provided.")

        return target_ids, validated_custom_ids


class ContextService:
    """Service class for chat context operations"""

    @staticmethod
    def initialize_user_context(user_id: int) -> None:
        """Initialize chat history for a user with system prompt"""
        if user_id not in CHAT_HISTORY:
            bot_name = random.choice(FEMALE_NAMES)
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
            CHAT_HISTORY[user_id] = [{
                "role": "system",
                "content": system_prompt,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]

    @staticmethod
    def add_context_instruction(user_id: int, instruction: str) -> None:
        """Add instruction to user's context"""
        ContextService.initialize_user_context(user_id)
        CHAT_HISTORY[user_id].append({
            "role": "system",
            "content": instruction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })


def handle_common_errors(func):
    """Decorator for common error handling"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.TimeoutError:
            logger.error(f"Timeout in {func.__name__}")
            return HTMLResponse(content="Request timeout", status_code=504)
        except ValueError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            return HTMLResponse(content=f"Validation error: {str(e)}", status_code=400)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return HTMLResponse(content="Internal server error", status_code=500)

    return wrapper


def validate_template_exists(template_name: str) -> None:
    """Validate that required template exists"""
    template_path = os.path.join("templates", template_name)
    if not os.path.exists(template_path):
        logger.error(f"{template_name} not found in templates directory")
        raise FileNotFoundError(f"Template {template_name} not found")


def init_web_routes(app: FastAPI, client, client_gpt):
    """Initialize all web routes"""
    logger.info("Starting initialization of web routes")

    # Check if templates directory exists
    if not os.path.exists("templates"):
        logger.error("Templates directory not found")
        raise FileNotFoundError("Templates directory not found")

    # Initialize services
    telegram_service = TelegramService(client)

    # Reset chat history if invalid data detected
    try:
        logger.info("Checking SQLite chat history")
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            c.execute("SELECT user_id, history FROM chat_history")
            for row in c.fetchall():
                try:
                    history = json.loads(row[1])
                    if not all(
                            isinstance(msg, dict) and "role" in msg and "content" in msg and "timestamp" in msg
                            for msg in history
                    ):
                        logger.warning(f"Invalid history format in SQLite for user {row[0]}, resetting chat history")
                        reset_chat_history()
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in SQLite for user {row[0]}, resetting chat history")
                    reset_chat_history()
                    break
        logger.info("SQLite chat history check completed")
    except Exception as e:
        logger.error(f"Error checking SQLite chat history: {e}")
        reset_chat_history()

    @app.get("/", response_class=HTMLResponse)
    @handle_common_errors
    async def gpt_message_form(request: Request):
        """Main page with message form"""
        logger.info("Handling GET / request")
        validate_template_exists("index.html")

        logger.info("Fetching user list")
        try:
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("Timeout while fetching user list")
            users = []

        logger.info(f"Users retrieved: {len(users)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "messages": [],
            "selected_user_ids": [],
            "preview_text": None,
            "error": "Unable to fetch user list due to timeout" if not users else None
        })

    @app.post("/", response_class=HTMLResponse)
    @handle_common_errors
    async def send_gpt_message(
            request: Request,
            user_ids: List[str] = Form(default=[]),
            custom_user_id: str = Form(default="")
    ):
        """Send GPT-generated message to selected users"""
        form_data = await request.form()
        logger.info(f"Send message request: user_ids={user_ids}, custom_user_id={custom_user_id}")

        # Validate and process user IDs
        target_ids, _ = await UserService.validate_and_process_user_ids(
            user_ids, custom_user_id, telegram_service
        )

        # Validate instruction
        instruction = form_data.get("instruction", "").strip()
        if not instruction:
            raise ValueError("Please provide an instruction for ChatGPT.")

        # Send messages to all target users
        for target_id in target_ids:
            generated_message = await ask_chatgpt(target_id, instruction, client, client_gpt)
            await telegram_service.send_message_with_typing(target_id, generated_message)

        save_data()  # Save auto-reply state after sending messages
        users = await asyncio.wait_for(get_dialog_user_list(client), timeout=REQUEST_TIMEOUT)

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

    @app.post("/get-messages", response_class=HTMLResponse)
    @handle_common_errors
    async def get_messages(
            request: Request,
            user_ids: List[str] = Form(default=[]),
            custom_user_id: str = Form(default="")
    ):
        """Get last messages from selected users"""
        logger.info(f"Get messages request: user_ids={user_ids}, custom_user_id={custom_user_id}")

        # Validate and process user IDs
        target_ids, _ = await UserService.validate_and_process_user_ids(
            user_ids, custom_user_id, telegram_service
        )

        # Fetch messages
        messages = await asyncio.wait_for(get_last_messages(client, target_ids), timeout=REQUEST_TIMEOUT)
        logger.info(f"Messages retrieved for {len(target_ids)} users")

        users = await asyncio.wait_for(get_dialog_user_list(client), timeout=REQUEST_TIMEOUT)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": messages,
            "preview_text": None
        })

    @app.post("/toggle-auto-reply", response_class=HTMLResponse)
    @handle_common_errors
    async def toggle_auto_reply(
            request: Request,
            user_ids: List[str] = Form(default=[]),
            custom_user_id: str = Form(default="")
    ):
        """Toggle auto-reply for selected users"""
        logger.info(f"Toggle auto-reply request: user_ids={user_ids}, custom_user_id={custom_user_id}")

        # Validate and process user IDs
        target_ids, _ = await UserService.validate_and_process_user_ids(
            user_ids, custom_user_id, telegram_service
        )

        # Determine whether to enable or disable auto-reply
        should_enable = any(target_id not in auto_reply_users for target_id in target_ids)

        for target_id in target_ids:
            if should_enable:
                auto_reply_users.add(target_id)
                AUTO_REPLY_STATUS.pop(target_id, None)  # Clear disable status when enabling
            else:
                auto_reply_users.discard(target_id)
                AUTO_REPLY_STATUS.pop(target_id, None)  # Clear disable status when disabling

        save_data()  # Save auto-reply state after toggling
        logger.info(f"Auto-reply toggled: should_enable={should_enable}, auto_reply_users={list(auto_reply_users)}")

        users = await asyncio.wait_for(get_dialog_user_list(client), timeout=REQUEST_TIMEOUT)
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

    @app.post("/preview-response", response_class=HTMLResponse)
    @handle_common_errors
    async def preview_response(
            request: Request,
            user_ids: List[str] = Form(default=[]),
            custom_user_id: str = Form(default="")
    ):
        """Preview GPT response without sending"""
        form_data = await request.form()
        logger.info(f"Preview response request: user_ids={user_ids}, custom_user_id={custom_user_id}")

        # Validate and process user IDs
        target_ids, _ = await UserService.validate_and_process_user_ids(
            user_ids, custom_user_id, telegram_service
        )

        # Validate instruction
        instruction = form_data.get("instruction", "").strip()
        if not instruction:
            raise ValueError("Please provide an instruction for ChatGPT.")

        # Generate preview for each user
        preview_texts = {}
        for target_id in target_ids:
            preview_texts[target_id] = await ask_chatgpt(
                target_id, instruction, client, client_gpt, send_message=False
            )

        users = await asyncio.wait_for(get_dialog_user_list(client), timeout=REQUEST_TIMEOUT)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "users": users,
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "messages": [],
            "preview_text": preview_texts
        })

    @app.post("/send-preview", response_class=HTMLResponse)
    @handle_common_errors
    async def send_preview(
            request: Request,
            user_ids: List[str] = Form(default=[]),
            custom_user_id: str = Form(default=""),
            preview_text: str = Form(...)
    ):
        """Send previewed text to selected users"""
        logger.info(f"Send preview request: user_ids={user_ids}, custom_user_id={custom_user_id}")

        # Validate and process user IDs
        target_ids, _ = await UserService.validate_and_process_user_ids(
            user_ids, custom_user_id, telegram_service
        )

        if not preview_text.strip():
            raise ValueError("No preview text available to send.")

        # Send preview text to all target users
        for target_id in target_ids:
            await telegram_service.send_message_with_typing(target_id, preview_text)

        save_data()  # Save auto-reply state after sending preview
        users = await asyncio.wait_for(get_dialog_user_list(client), timeout=REQUEST_TIMEOUT)

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

    @app.get("/keywords", response_class=HTMLResponse)
    @handle_common_errors
    async def keywords_form(request: Request):
        """Keywords management form"""
        validate_template_exists("keywords.html")

        return templates.TemplateResponse("keywords.html", {
            "request": request,
            "keywords": AUTO_REPLY_DISABLE_KEYWORDS,
            "notification_user_id": NOTIFICATION_USER_ID,
            "success": None,
            "error": None
        })

    @app.post("/keywords", response_class=HTMLResponse)
    @handle_common_errors
    async def update_keywords(
            request: Request,
            keywords: str = Form(...),
            notification_user_id: str = Form(default="")
    ):
        """Update keywords and notification settings"""
        logger.info(f"Update keywords request: keywords={keywords}, notification_user_id={notification_user_id}")

        # Process keywords
        new_keywords = [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]
        if not new_keywords:
            raise ValueError("At least one keyword must be provided.")

        # Validate notification user ID
        new_notification_user_id = notification_user_id.strip() or None
        if new_notification_user_id:
            try:
                new_notification_user_id = int(new_notification_user_id)
                if not await telegram_service.validate_user_entity(new_notification_user_id):
                    raise ValueError(f"Invalid Telegram user ID: {new_notification_user_id}")
            except (ValueError, TypeError):
                raise ValueError(f"Invalid Telegram user ID format: {notification_user_id}")

        # Update global state
        AUTO_REPLY_DISABLE_KEYWORDS.clear()
        AUTO_REPLY_DISABLE_KEYWORDS.extend(new_keywords)
        global NOTIFICATION_USER_ID
        NOTIFICATION_USER_ID = new_notification_user_id

        # Save changes
        save_keywords(new_keywords)
        save_config(new_notification_user_id)

        return templates.TemplateResponse("keywords.html", {
            "request": request,
            "keywords": AUTO_REPLY_DISABLE_KEYWORDS,
            "notification_user_id": NOTIFICATION_USER_ID,
            "success": "Keywords and notification user ID updated successfully.",
            "error": None
        })

    @app.get("/context", response_class=HTMLResponse)
    @handle_common_errors
    async def update_context_form(request: Request):
        """Context management form"""
        logger.info("Handling GET /context request")
        validate_template_exists("context.html")

        users = await asyncio.wait_for(get_dialog_user_list(client), timeout=REQUEST_TIMEOUT)
        logger.info(f"Retrieved {len(users)} users for /context")

        return templates.TemplateResponse("context.html", {
            "request": request,
            "users": users,
            "context_preview_text": None,
            "success": None,
            "error": None
        })

    @app.post("/update-context", response_class=HTMLResponse)
    @handle_common_errors
    async def update_context(
            request: Request,
            user_ids: List[str] = Form(default=[]),
            custom_user_id: str = Form(default=""),
            instruction: str = Form(default="")
    ):
        """Update context for selected users"""
        logger.info(f"Update context request: user_ids={user_ids}, custom_user_id={custom_user_id}")

        # Validate and process user IDs
        target_ids, _ = await UserService.validate_and_process_user_ids(
            user_ids, custom_user_id, telegram_service
        )

        if not instruction.strip():
            raise ValueError("Please provide an instruction to update context.")

        # Update context for each user
        for target_id in target_ids:
            ContextService.add_context_instruction(target_id, instruction)

        save_data()  # Save changes

        users = await asyncio.wait_for(get_dialog_user_list(client), timeout=REQUEST_TIMEOUT)
        return templates.TemplateResponse("context.html", {
            "request": request,
            "users": users,
            "success": "Context updated successfully for selected users.",
            "sent_text": instruction,
            "selected_user_ids": target_ids,
            "custom_user_id": custom_user_id,
            "context_preview_text": None
        })

    @app.get("/view-context/{user_id}", response_class=HTMLResponse)
    @handle_common_errors
    async def view_context(request: Request, user_id: int):
        """View context history for specific user"""
        validate_template_exists("view_context.html")

        # Get context history for user_id
        context_history = CHAT_HISTORY.get(user_id, [])

        return templates.TemplateResponse("view_context.html", {
            "request": request,
            "user_id": user_id,
            "context_history": context_history,
            "error": None
        })

    logger.info("Web routes initialization completed")