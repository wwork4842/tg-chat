import json
import sqlite3
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict
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

# Initialize templates
templates = Jinja2Templates(directory="templates")


def init_web_routes(app, tg_client, gpt_client):
    @app.get("/keywords", response_class=HTMLResponse)
    async def keywords_page(request: Request):
        # Load config.json
        config = {}
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        # Load keywords.json
        keywords = []
        if os.path.exists("keywords.json"):
            with open("keywords.json", "r", encoding="utf-8") as f:
                keywords = json.load(f)
        return templates.TemplateResponse(
            "keywords.html",
            {
                "request": request,
                "config": config,
                "keywords": keywords
            }
        )
# Check if templates directory exists
if not os.path.exists("templates"):
    logger.error("Templates directory not found")
    raise FileNotFoundError("Templates directory not found")


def init_web_routes(app: FastAPI, client, client_gpt):
    logger.info("Starting initialization of web routes")

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
                            isinstance(msg, dict) and "role" in msg and "content" in msg and "timestamp" in msg for msg
                            in history):
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

    logger.info("Registering route: /")
    @app.get("/", response_class=HTMLResponse)
    async def gpt_message_form(request: Request):
        try:
            logger.info("Handling GET / request")
            if not os.path.exists(os.path.join("templates", "index.html")):
                logger.error("index.html not found in templates directory")
                return HTMLResponse(content="Template index.html not found", status_code=500)
            logger.info("Fetching user list")
            try:
                users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            except asyncio.TimeoutError:
                logger.error("Timeout while fetching user list")
                users = []
                logger.info("Returning empty user list due to timeout")
            logger.info(f"Users retrieved: {len(users)}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "users": users,
                "messages": [],
                "selected_user_ids": [],
                "preview_text": None,
                "error": "Unable to fetch user list due to timeout" if not users else None
            })
        except Exception as e:
            logger.error(f"Error in gpt_message_form: {e}")
            return HTMLResponse(content=f"Internal server error: {str(e)}", status_code=500)

    logger.info("Registering route: / (POST)")
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
                    entity = await asyncio.wait_for(client.get_entity(target_id), timeout=5.0)
                    if not isinstance(entity, User):
                        logger.warning(f"ID is not a user: {target_id}")
                        if target_id in custom_ids:
                            custom_ids.remove(target_id)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout while fetching entity for ID: {target_id}")
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
                generated_message = await ask_chatgpt(target_id, instruction, client, client_gpt)
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

            save_data()  # Save auto-reply state after sending messages
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
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
        except asyncio.TimeoutError:
            logger.error("Timeout while processing send_gpt_message")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error in send_gpt_message: {e}")
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "users": users,
                "error": str(e),
                "selected_user_ids": user_ids_int,
                "custom_user_id": custom_user_id,
                "messages": [],
                "preview_text": None
            })

    logger.info("Registering route: /get-messages")
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
                    entity = await asyncio.wait_for(client.get_entity(target_id), timeout=5.0)
                    if not isinstance(entity, User):
                        logger.warning(f"ID is not a user: {target_id}")
                        if target_id in custom_ids:
                            custom_ids.remove(target_id)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout while fetching entity for ID: {target_id}")
                    if target_id in custom_ids:
                        custom_ids.remove(target_id)
                except PeerIdInvalidError:
                    logger.warning(f"Invalid Telegram user ID: {target_id}")
                    if target_id in custom_ids:
                        custom_ids.remove(target_id)

            target_ids = list(set(user_ids_int + custom_ids))
            if not target_ids:
                raise ValueError("No valid user IDs provided.")

            messages = await asyncio.wait_for(get_last_messages(client, target_ids), timeout=15.0)
            logger.info(f"Messages retrieved for {len(target_ids)} users")

            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "users": users,
                "selected_user_ids": target_ids,
                "custom_user_id": custom_user_id,
                "messages": messages,
                "preview_text": None
            })
        except asyncio.TimeoutError:
            logger.error("Timeout while processing get_messages")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error in get_messages: {e}")
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "users": users,
                "error": str(e),
                "selected_user_ids": user_ids_int,
                "custom_user_id": custom_user_id,
                "messages": [],
                "preview_text": None
            })

    logger.info("Registering route: /toggle-auto-reply")
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
                    entity = await asyncio.wait_for(client.get_entity(target_id), timeout=5.0)
                    if not isinstance(entity, User):
                        logger.warning(f"ID is not a user: {target_id}")
                        if target_id in custom_ids:
                            custom_ids.remove(target_id)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout while fetching entity for ID: {target_id}")
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
            save_data()  # Save auto-reply state after toggling
            logger.info(f"Auto-reply toggled: should_enable={should_enable}, auto_reply_users={list(auto_reply_users)}")

            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
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
        except asyncio.TimeoutError:
            logger.error("Timeout while processing toggle_auto_reply")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error in toggle_auto_reply: {e}")
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "users": users,
                "error": str(e),
                "selected_user_ids": user_ids_int,
                "custom_user_id": custom_user_id,
                "messages": [],
                "preview_text": None
            })

    logger.info("Registering route: /preview-response")
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
                    entity = await asyncio.wait_for(client.get_entity(target_id), timeout=5.0)
                    if not isinstance(entity, User):
                        logger.warning(f"ID is not a user: {target_id}")
                        if target_id in custom_ids:
                            custom_ids.remove(target_id)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout while fetching entity for ID: {target_id}")
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
                preview_texts[target_id] = await ask_chatgpt(target_id, instruction, client, client_gpt,
                                                             send_message=False)

            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "users": users,
                "selected_user_ids": target_ids,
                "custom_user_id": custom_user_id,
                "messages": [],
                "preview_text": preview_texts
            })
        except asyncio.TimeoutError:
            logger.error("Timeout while processing preview_response")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error in preview_response: {e}")
            try:
                users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "users": users,
                    "error": str(e),
                    "selected_user_ids": user_ids_int,
                    "custom_user_id": custom_user_id,
                    "messages": [],
                    "preview_text": None
                })
            except Exception as template_e:
                logger.error(f"Error rendering template in preview_response: {template_e}")
                return HTMLResponse(content="Internal server error", status_code=500)

    logger.info("Registering route: /send-preview")
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

            save_data()  # Save auto-reply state after sending preview
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
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
        except asyncio.TimeoutError:
            logger.error("Timeout while processing send_preview")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error in send_preview: {e}")
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "users": users,
                "error": str(e),
                "selected_user_ids": user_ids_int,
                "custom_user_id": custom_user_id,
                "messages": [],
                "preview_text": None
            })

    logger.info("Registering route: /keywords")
    @app.get("/keywords", response_class=HTMLResponse)
    async def keywords_form(request: Request):
        try:
            if not os.path.exists(os.path.join("templates", "keywords.html")):
                logger.error("keywords.html not found in templates directory")
                return HTMLResponse(content="Template keywords.html not found", status_code=500)
            return templates.TemplateResponse("keywords.html", {
                "request": request,
                "keywords": AUTO_REPLY_DISABLE_KEYWORDS,
                "notification_user_id": NOTIFICATION_USER_ID,
                "success": None,
                "error": None
            })
        except asyncio.TimeoutError:
            logger.error("Timeout while processing keywords_form")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error in keywords_form: {e}")
            return HTMLResponse(content=f"Internal server error: {str(e)}", status_code=500)

    logger.info("Registering route: /keywords (POST)")
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
                    entity = await asyncio.wait_for(client.get_entity(new_notification_user_id), timeout=5.0)
                    if not isinstance(entity, User):
                        raise ValueError(f"ID {new_notification_user_id} does not correspond to a user.")
                except asyncio.TimeoutError:
                    raise ValueError(f"Timeout while validating Telegram user ID: {new_notification_user_id}")
                except (ValueError, PeerIdInvalidError):
                    raise ValueError(f"Invalid Telegram user ID: {new_notification_user_id}")

            AUTO_REPLY_DISABLE_KEYWORDS.clear()
            AUTO_REPLY_DISABLE_KEYWORDS.extend(new_keywords)
            global NOTIFICATION_USER_ID
            NOTIFICATION_USER_ID = new_notification_user_id
            save_keywords(new_keywords)  # Ensure keywords are saved
            save_config(new_notification_user_id)  # Ensure config is saved
            return templates.TemplateResponse("keywords.html", {
                "request": request,
                "keywords": AUTO_REPLY_DISABLE_KEYWORDS,
                "notification_user_id": NOTIFICATION_USER_ID,
                "success": "Keywords and notification user ID updated successfully.",
                "error": None
            })
        except asyncio.TimeoutError:
            logger.error("Timeout while processing update_keywords")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error updating keywords or notification user ID: {e}")
            return templates.TemplateResponse("keywords.html", {
                "request": request,
                "keywords": AUTO_REPLY_DISABLE_KEYWORDS,
                "notification_user_id": NOTIFICATION_USER_ID,
                "success": None,
                "error": str(e)
            })

    logger.info("Registering route: /context")
    @app.get("/context", response_class=HTMLResponse)
    async def update_context_form(request: Request):
        logger.info("Handling GET /context request")
        try:
            if not os.path.exists(os.path.join("templates", "context.html")):
                logger.error("context.html not found in templates directory")
                return HTMLResponse(content="Template context.html not found", status_code=500)
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            logger.info(f"Retrieved {len(users)} users for /context")
            return templates.TemplateResponse("context.html", {
                "request": request,
                "users": users,
                "context_preview_text": None,
                "success": None,
                "error": None
            })
        except asyncio.TimeoutError:
            logger.error("Timeout while processing update_context_form")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error in update_context_form: {e}")
            return HTMLResponse(content=f"Internal server error: {str(e)}", status_code=500)

    logger.info("Registering route: /update-context")
    @app.post("/update-context", response_class=HTMLResponse)
    async def update_context(
            request: Request,
            user_ids: List[str] = Form(default=[]),
            custom_user_id: str = Form(default=""),
            instruction: str = Form(default="")
    ):
        try:
            form_data = await request.form()
            logger.info(
                f"Update context request: user_ids={user_ids}, custom_user_id={custom_user_id}, instruction={instruction}, form_data={dict(form_data)}")

            # Validate user_ids
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

            # Validate Telegram user IDs
            for target_id in custom_ids:
                try:
                    entity = await asyncio.wait_for(client.get_entity(target_id), timeout=5.0)
                    if not isinstance(entity, User):
                        logger.warning(f"ID is not a user: {target_id}")
                        if target_id in custom_ids:
                            custom_ids.remove(target_id)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout while fetching entity for ID: {target_id}")
                    if target_id in custom_ids:
                        custom_ids.remove(target_id)
                except PeerIdInvalidError:
                    logger.warning(f"Invalid Telegram user ID: {target_id}")
                    if target_id in custom_ids:
                        custom_ids.remove(target_id)

            target_ids = list(set(user_ids_int + custom_ids))
            if not target_ids:
                raise ValueError("No valid user IDs provided.")

            if not instruction.strip():
                raise ValueError("Please provide an instruction to update context.")

            # Update CHAT_HISTORY for each user_id
            for target_id in target_ids:
                if target_id not in CHAT_HISTORY:
                    # Initialize history with system prompt
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
                    CHAT_HISTORY[target_id] = [{
                        "role": "system",
                        "content": system_prompt,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }]

                # Add instruction as system message
                CHAT_HISTORY[target_id].append({
                    "role": "system",
                    "content": instruction,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            # Save changes
            save_data()

            # Get user list for display
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            return templates.TemplateResponse("context.html", {
                "request": request,
                "users": users,
                "success": "Context updated successfully for selected users.",
                "sent_text": instruction,
                "selected_user_ids": target_ids,
                "custom_user_id": custom_user_id,
                "context_preview_text": None
            })
        except asyncio.TimeoutError:
            logger.error("Timeout while processing update_context")
            return HTMLResponse(content="Timeout while processing request", status_code=504)
        except Exception as e:
            logger.error(f"Error in update_context: {e}")
            users = await asyncio.wait_for(get_dialog_user_list(client), timeout=15.0)
            return templates.TemplateResponse("context.html", {
                "request": request,
                "users": users,
                "error": str(e),
                "selected_user_ids": user_ids_int,
                "custom_user_id": custom_user_id,
                "context_preview_text": None
            })

    logger.info("Registering route: /view-context/{user_id}")
    @app.get("/view-context/{user_id}", response_class=HTMLResponse)
    async def view_context(request: Request, user_id: int):
        try:
            if not os.path.exists(os.path.join("templates", "view_context.html")):
                logger.error("view_context.html not found in templates directory")
                return HTMLResponse(content="Template view_context.html not found", status_code=500)

            # Get context history for user_id
            context_history = CHAT_HISTORY.get(user_id, [])

            return templates.TemplateResponse("view_context.html", {
                "request": request,
                "user_id": user_id,
                "context_history": context_history,
                "error": None
            })
        except Exception as e:
            logger.error(f"Error in view_context: {e}")
            return templates.TemplateResponse("view_context.html", {
                "request": request,
                "user_id": user_id,
                "context_history": [],
                "error": str(e)
            })

    logger.info("Web routes initialization completed")