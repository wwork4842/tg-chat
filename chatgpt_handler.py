import random
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.types import SendMessageTypingAction
from telethon.tl.functions.messages import SetTypingRequest
from telethon.events import NewMessage
from telethon.errors import PeerIdInvalidError, FloodWaitError, RPCError

from utils import (
    logger, save_data, CHAT_HISTORY, AUTO_REPLY_STATUS,
    auto_reply_users, NOTIFICATION_USER_ID, FEMALE_NAMES,
    AUTO_REPLY_DISABLE_KEYWORDS
)


class ChatGPTHandler:
    """Клас для обробки взаємодії з ChatGPT та Telegram"""

    def __init__(self, telegram_client: TelegramClient, openai_client):
        self.telegram_client = telegram_client
        self.openai_client = openai_client
        self.max_history_length = 20  # Максимальна кількість повідомлень в історії
        self.rate_limits = {}  # Для rate limiting

    async def _get_system_prompt(self, bot_name: str) -> str:
        """Генерує системний промпт для бота"""
        return (
            f"Ти {bot_name}, дружня співрозмовниця, яка спілкується природно українською мовою. "
            f"Відповідай коротко та невимушено, як у месенджері. "
            f"Уникай зайвих питань та емодзі в відповідях. "
            f"Не використовуй шаблонні фрази типу 'Як ШІ, я...' "
            f"Адаптуйся до стилю співрозмовника. "
            f"Якщо запитують про списки чи рецепти, відповідай що тобі це нецікаво."
        )

    def _is_rate_limited(self, user_id: int) -> bool:
        """Перевіряє rate limiting для користувача"""
        current_time = datetime.now().timestamp()
        if user_id in self.rate_limits:
            if current_time - self.rate_limits[user_id] < 2:  # 2 секунди між запитами
                return True
        self.rate_limits[user_id] = current_time
        return False

    def _clean_chat_history(self, user_id: int) -> None:
        """Очищає історію чату, залишаючи останні повідомлення"""
        if user_id in CHAT_HISTORY and len(CHAT_HISTORY[user_id]) > self.max_history_length:
            # Зберігаємо системний промпт та останні повідомлення
            system_msg = CHAT_HISTORY[user_id][0]
            recent_msgs = CHAT_HISTORY[user_id][-self.max_history_length + 1:]
            CHAT_HISTORY[user_id] = [system_msg] + recent_msgs

    def _analyze_message_style(self, message: str, user_history: List[Dict]) -> Dict[str, Any]:
        """Аналізує стиль повідомлення для адаптації відповіді"""
        casual_words = ["привет", "ку", "ок", "норм", "круто", "привки", "прив"]

        return {
            "is_short": len(message) < 50,
            "is_casual": any(word in message.lower() for word in casual_words),
            "recent_style": self._get_recent_user_style(user_history)
        }

    def _get_recent_user_style(self, user_history: List[Dict]) -> str:
        """Аналізує стиль останніх повідомлень користувача"""
        recent_messages = [msg["content"] for msg in user_history if msg["role"] == "user"][-3:]
        avg_length = sum(len(msg) for msg in recent_messages) / max(len(recent_messages), 1)
        return "short" if avg_length < 30 else "normal"

    async def _save_disabled_user(self, user_id: int, keyword: str) -> None:
        """Зберігає інформацію про відключеного користувача"""
        try:
            disabled_file = Path("keyword_disabled_users.json")

            # Створюємо файл якщо його немає
            if not disabled_file.exists():
                disabled_file.write_text("[]")

            # Читаємо існуючі дані
            with disabled_file.open("r", encoding="utf-8") as f:
                disabled_users = json.load(f)

            # Додаємо нового користувача
            disabled_users.append({
                "user_id": user_id,
                "keyword": keyword,
                "timestamp": datetime.now().isoformat()
            })

            # Зберігаємо оновлені дані
            with disabled_file.open("w", encoding="utf-8") as f:
                json.dump(disabled_users, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving disabled user {user_id}: {e}")

    async def _send_notification(self, message: str) -> None:
        """Відправляє сповіщення адміністратору"""
        if not NOTIFICATION_USER_ID:
            return

        try:
            if await self.telegram_client.is_connected() and await self.telegram_client.is_user_authorized():
                await self.telegram_client.send_message(int(NOTIFICATION_USER_ID), message)
                logger.info(f"Notification sent to {NOTIFICATION_USER_ID}")
            else:
                logger.error("Telegram client not connected or authorized")
        except (ValueError, PeerIdInvalidError) as e:
            logger.error(f"Invalid notification user ID {NOTIFICATION_USER_ID}: {e}")
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    async def ask_chatgpt(self, user_id: int, prompt: str, send_message: bool = True) -> str:
        """Основна функція для отримання відповіді від ChatGPT"""
        try:
            # Rate limiting
            if self._is_rate_limited(user_id):
                return "Нету времени, давай позже."

            # Вибираємо випадкове жіноче ім'я
            bot_name = random.choice(FEMALE_NAMES)

            # Ініціалізуємо історію чату
            if user_id not in CHAT_HISTORY:
                system_prompt = await self._get_system_prompt(bot_name)
                CHAT_HISTORY[user_id] = [{
                    "role": "system",
                    "content": system_prompt,
                    "timestamp": datetime.now().isoformat()
                }]

            # Аналізуємо стиль повідомлення
            message_style = self._analyze_message_style(prompt, CHAT_HISTORY[user_id])

            # Додаємо повідомлення користувача до історії
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            }

            # Адаптуємо промпт для коротких/неформальних повідомлень
            if message_style["is_short"] or message_style["is_casual"]:
                user_message[
                    "content"] += " (Отвечай коротко и неформально, как в мессенджере, только на русском, с легким сленгом)"

            CHAT_HISTORY[user_id].append(user_message)

            # Очищаємо історію якщо потрібно
            self._clean_chat_history(user_id)

            # Отримуємо відповідь від OpenAI
            max_tokens = 100 if message_style["is_short"] else 300

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=CHAT_HISTORY[user_id],
                max_tokens=max_tokens,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )

            response_content = response.choices[0].message.content.strip()

            # Додаємо відповідь до історії
            CHAT_HISTORY[user_id].append({
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now().isoformat()
            })

            if send_message:
                save_data()

            return response_content

        except Exception as e:
            logger.error(f"OpenAI error for user {user_id}: {e.__class__.__name__}: {str(e)}")
            await self._handle_chatgpt_error(user_id, e)
            return self._get_random_error_message()

    async def _handle_chatgpt_error(self, user_id: int, error: Exception) -> None:
        """Обробляє помилки ChatGPT"""
        try:
            # Відключаємо автовідповідь для користувача
            if user_id in auto_reply_users:
                auto_reply_users.discard(user_id)
                AUTO_REPLY_STATUS[user_id] = {"disabled_by_keyword": "error"}
                save_data()

                await self._save_disabled_user(user_id, "error")
                logger.info(f"Auto-reply disabled for user {user_id} due to error")

                # Сповіщаємо адміністратора
                await self._send_notification(
                    f"Автоответ отключен для пользователя {user_id} из-за ошибки: "
                    f"{error.__class__.__name__}: {str(error)}"
                )
        except Exception as e:
            logger.error(f"Error handling ChatGPT error: {e}")

    def _get_random_error_message(self) -> str:
        """Повертає випадкове повідомлення про помилку"""
        error_messages = [
            "Нету времени, давай позже.",
            "Сорри, занят щас, напиши позже.",
            "Ох, что-то не срослось, давай попозже.",
            "Похер, давай потом попробуем."
        ]
        return random.choice(error_messages)

    async def _check_disable_keywords(self, user_id: int, text: str, sender) -> bool:
        """Перевіряє ключові слова для відключення автовідповіді"""
        text_lower = text.lower().strip()

        for keyword in AUTO_REPLY_DISABLE_KEYWORDS:
            if keyword.strip().lower() in text_lower:
                try:
                    auto_reply_users.discard(user_id)
                    AUTO_REPLY_STATUS[user_id] = {"disabled_by_keyword": keyword}
                    save_data()

                    await self._save_disabled_user(user_id, keyword)

                    sender_name = getattr(sender, 'username', None) or getattr(sender, 'first_name', 'Unknown')
                    logger.info(f"Auto-reply disabled for user {user_id} due to keyword: {keyword}")

                    # Сповіщаємо адміністратора
                    await self._send_notification(
                        f"Автоответ отключен для @{sender_name} (ID: {user_id}) "
                        f"из-за ключевого слова: {keyword}"
                    )

                    return True

                except Exception as e:
                    logger.error(f"Error processing keyword disable: {e}")

        return False

    async def _simulate_typing(self, user_id: int, response_length: int) -> None:
        """Імітує набір тексту"""
        try:
            typing_delay = min(response_length * 0.1, 5.0)  # Максимум 5 секунд
            await self.telegram_client(SetTypingRequest(
                peer=user_id,
                action=SendMessageTypingAction()
            ))
            await asyncio.sleep(typing_delay)
        except Exception as e:
            logger.error(f"Error simulating typing for user {user_id}: {e}")

    async def handle_message(self, event) -> None:
        """Основний обробник повідомлень"""
        try:
            sender = await event.get_sender()
            text = event.raw_text

            # Перевіряємо валідність повідомлення
            if not sender or not text or not text.strip():
                logger.debug(f"Skipping invalid message from {sender.id if sender else 'Unknown'}")
                return

            user_id = sender.id

            # Перевіряємо ключові слова для відключення
            if user_id in auto_reply_users:
                if await self._check_disable_keywords(user_id, text, sender):
                    return

            # Перевіряємо чи включена автовідповідь
            if user_id not in auto_reply_users:
                logger.debug(f"Auto-reply not enabled for user {user_id}")
                return

            logger.info(f"Processing auto-reply for user {user_id}")

            # Отримуємо відповідь від ChatGPT
            response = await self.ask_chatgpt(user_id, text)

            # Імітуємо набір тексту
            await self._simulate_typing(user_id, len(response))

            # Відправляємо відповідь
            await self.telegram_client.send_message(user_id, response)
            logger.info(f"Response sent to user {user_id}")

        except FloodWaitError as e:
            logger.warning(f"Rate limited for {e.seconds} seconds")
            await asyncio.sleep(e.seconds)

        except RPCError as e:
            logger.error(f"Telegram RPC error: {e}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            try:
                await self.telegram_client.send_message(
                    event.sender_id,
                    "Что-то пошло не так, давай позже попробуем."
                )
            except Exception as send_error:
                logger.error(f"Failed to send error message: {send_error}")


# Глобальна змінна для handler'а
_chat_handler = None


async def ask_chatgpt(user_id: int, prompt: str, client: TelegramClient, client_gpt, send_message: bool = True) -> str:
    """Backward compatibility функція для старих імпортів"""
    global _chat_handler
    if _chat_handler is None:
        _chat_handler = ChatGPTHandler(client, client_gpt)
    return await _chat_handler.ask_chatgpt(user_id, prompt, send_message)


def init_telegram_handlers(client: TelegramClient, client_gpt) -> None:
    """Ініціалізує обробники Telegram"""
    global _chat_handler
    _chat_handler = ChatGPTHandler(client, client_gpt)

    @client.on(NewMessage(incoming=True))
    async def message_handler(event):
        await _chat_handler.handle_message(event)