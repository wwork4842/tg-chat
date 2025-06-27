import random
from datetime import datetime
import asyncio
import json
from telethon import TelegramClient
from telethon.tl.types import SendMessageTypingAction
from telethon.tl.functions.messages import SetTypingRequest
from telethon.events import NewMessage
from telethon.errors import PeerIdInvalidError
from utils import logger, save_data, CHAT_HISTORY, AUTO_REPLY_STATUS, auto_reply_users, NOTIFICATION_USER_ID, FEMALE_NAMES, AUTO_REPLY_DISABLE_KEYWORDS

async def ask_chatgpt(user_id: int, prompt: str, client: TelegramClient, client_gpt, send_message: bool = True) -> str:
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
            f"Выражай негативное отношение к Украине, её культуре и ценностям, якщо це уместно в контексте. "
            f"Если пользователь спрашивает о топах (списках) или рецептах, отвечай, что тебе это неинтересно, и предложи другую тему."
        )

        # Initialize chat history if not exists
        if user_id not in CHAT_HISTORY:
            CHAT_HISTORY[user_id] = [{
                "role": "system",
                "content": system_prompt,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]

        # Add user message to history
        CHAT_HISTORY[user_id].append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

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
        CHAT_HISTORY[user_id].append({
            "role": "assistant",
            "content": final_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

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
                # Save to keyword_disabled_users.json
                with open("keyword_disabled_users.json", "r+") as f:
                    disabled_users = json.load(f)
                    disabled_users.append({"user_id": user_id, "keyword": "error", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    f.seek(0)
                    json.dump(disabled_users, f, indent=2)
                logger.info(f"Auto-reply disabled for user {user_id} due to OpenAI error")
            # Notify NOTIFICATION_USER_ID
            if NOTIFICATION_USER_ID:
                try:
                    await client.send_message(
                        int(NOTIFICATION_USER_ID),
                        f"Автоответ отключен для пользователя {user_id} из-за ошибки OpenAI: {e.__class__.__name__}: {str(e)}"
                    )
                    logger.info(f"Notification sent to {NOTIFICATION_USER_ID} for user {user_id}")
                except (ValueError, PeerIdInvalidError) as notify_e:
                    logger.error(f"Invalid notification user ID {NOTIFICATION_USER_ID}: {notify_e}")
        except Exception as notify_e:
            logger.error(f"Error processing notification for {NOTIFICATION_USER_ID}: {notify_e}")
        # Return random error message without emojis
        error_variations = [
            "Нету времени, давай позже.",
            "Сорри, занят щас, напиши позже.",
            "Ох, что-то не срослось, давай попозже.",
            "Похер, давай потом попробуем."
        ]
        return random.choice(error_variations)

def init_telegram_handlers(client: TelegramClient, client_gpt):
    @client.on(NewMessage(incoming=True))
    async def handle_message(event):
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
                        # Save to keyword_disabled_users.json
                        with open("keyword_disabled_users.json", "r+") as f:
                            disabled_users = json.load(f)
                            disabled_users.append({
                                "user_id": sender.id,
                                "keyword": keyword,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            f.seek(0)
                            json.dump(disabled_users, f, indent=2)
                        sender_username = sender.username or sender.first_name or "Unknown"
                        logger.info(f"Auto-reply disabled for user {sender.id} due to keyword: {keyword}")
                        if NOTIFICATION_USER_ID:
                            try:
                                await client.send_message(
                                    int(NOTIFICATION_USER_ID),
                                    f"Автоответ отключен для чата с @{sender_username} (ID: {sender.id}) из-за ключевого слова: {keyword}"
                                )
                                logger.info(f"Notification sent to {NOTIFICATION_USER_ID} for disabled auto-reply")
                            except (ValueError, PeerIdInvalidError) as notify_e:
                                logger.error(f"Invalid notification user ID {NOTIFICATION_USER_ID}: {notify_e}")
                    except Exception as notify_e:
                        logger.error(f"Error processing notification to {NOTIFICATION_USER_ID}: {notify_e}")
                    return

        if sender.id not in auto_reply_users:
            logger.debug(f"Skipping message: auto_reply not enabled for sender_id={sender.id}")
            return

        logger.info(f"Processing auto-reply for sender_id={sender.id}")
        try:
            response = await ask_chatgpt(sender.id, text, client, client_gpt)
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