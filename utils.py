import os
import json
import sqlite3
import logging
from typing import List, Dict
from telethon import TelegramClient
from telethon.tl.types import User
from dotenv import load_dotenv
import asyncio
from datetime import datetime

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

# Global state
auto_reply_users = set()
CHAT_HISTORY: Dict[int, List[Dict]] = {}
AUTO_REPLY_STATUS: Dict[int, Dict] = {}
AUTO_REPLY_DISABLE_KEYWORDS = []
NOTIFICATION_USER_ID = None


def load_config():
    global NOTIFICATION_USER_ID
    try:
        if os.path.exists(CONFIG["CONFIG_FILE"]):
            with open(CONFIG["CONFIG_FILE"], "r") as f:
                config_data = json.load(f)
                NOTIFICATION_USER_ID = config_data.get("notification_user_id", None)
                if NOTIFICATION_USER_ID is not None:
                    NOTIFICATION_USER_ID = int(NOTIFICATION_USER_ID)
                logger.info(f"Loaded config: notification_user_id={NOTIFICATION_USER_ID}")
                return NOTIFICATION_USER_ID
        logger.warning(f"Config file {CONFIG['CONFIG_FILE']} not found, using default None")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {CONFIG['CONFIG_FILE']}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None


def save_config(notification_user_id: str | None):
    try:
        config_data = {"notification_user_id": notification_user_id}
        with open(CONFIG["CONFIG_FILE"], "w") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Config saved: notification_user_id={notification_user_id}")
    except Exception as e:
        logger.error(f"Error saving config to {CONFIG['CONFIG_FILE']}: {e}")


def load_keywords():
    try:
        if os.path.exists(CONFIG["KEYWORDS_FILE"]):
            with open(CONFIG["KEYWORDS_FILE"], "r") as f:
                keywords = json.load(f)
                if isinstance(keywords, list):
                    return [keyword.strip().lower() for keyword in keywords]
        logger.warning(f"Keywords file {CONFIG['KEYWORDS_FILE']} not found, using default")
        return ["stop", "disable", "off"]  # Default keywords
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in keywords file {CONFIG['KEYWORDS_FILE']}: {e}")
        return ["stop", "disable", "off"]
    except Exception as e:
        logger.error(f"Error loading keywords: {e}")
        return ["stop", "disable", "off"]


def save_keywords(keywords: List[str]):
    try:
        with open(CONFIG["KEYWORDS_FILE"], "w") as f:
            json.dump(keywords, f, indent=2)
        logger.info(f"Keywords saved: {keywords}")
    except Exception as e:
        logger.error(f"Error saving keywords to {CONFIG['KEYWORDS_FILE']}: {e}")


def validate_env_vars():
    required_vars = ["OPENAI_API_KEY", "TG_API_ID", "TG_API_HASH", "TG_PHONE"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")


def save_session(client: TelegramClient):
    try:
        session_string = client.session.save()
        with open(CONFIG["SESSION_FILE"], "w") as f:
            f.write(session_string)
        logger.info(f"Session saved to {CONFIG['SESSION_FILE']}")
    except Exception as e:
        logger.error(f"Failed to save session: {e}")


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
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def reset_chat_history():
    try:
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM chat_history")
            conn.commit()
        logger.info("Chat history table reset")
    except Exception as e:
        logger.error(f"Error resetting chat history: {e}")


def load_data():
    global auto_reply_users, CHAT_HISTORY, AUTO_REPLY_STATUS, AUTO_REPLY_DISABLE_KEYWORDS, NOTIFICATION_USER_ID
    try:
        # Load auto-reply users
        if os.path.exists(CONFIG["AUTO_REPLY_FILE"]):
            with open(CONFIG["AUTO_REPLY_FILE"], "r") as f:
                auto_reply_users.update(set(json.load(f)))

        # Load from SQLite
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            c.execute("SELECT user_id, history FROM chat_history")
            for row in c.fetchall():
                try:
                    history = json.loads(row[1])
                    if isinstance(history, list) and all(
                            isinstance(msg, dict) and "role" in msg and "content" in msg and "timestamp" in msg for msg
                            in history):
                        CHAT_HISTORY[row[0]] = history
                    else:
                        logger.warning(f"Invalid history format for user {row[0]}, skipping")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in chat history for user {row[0]}")

            c.execute("SELECT user_id, disabled_by_keyword FROM auto_reply_status")
            AUTO_REPLY_STATUS.update(
                {row[0]: {"disabled_by_keyword": row[1]} for row in c.fetchall() if row[1] is not None})

        logger.info(
            f"Data loaded: {len(auto_reply_users)} auto-reply users, {len(CHAT_HISTORY)} chat history users, {len(AUTO_REPLY_STATUS)} auto-reply status entries")

        AUTO_REPLY_DISABLE_KEYWORDS = load_keywords()
        NOTIFICATION_USER_ID = load_config()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        CHAT_HISTORY.clear()
        AUTO_REPLY_STATUS.clear()


def save_data():
    try:
        # Save auto-reply users
        with open(CONFIG["AUTO_REPLY_FILE"], "w") as f:
            json.dump(list(auto_reply_users), f, indent=2)

        # Save to SQLite
        with sqlite3.connect(CONFIG["DB_PATH"], timeout=CONFIG["SQLITE_TIMEOUT"]) as conn:
            c = conn.cursor()
            for user_id, history in CHAT_HISTORY.items():
                try:
                    # Validate history format
                    if not all(
                            isinstance(msg, dict) and "role" in msg and "content" in msg and "timestamp" in msg for msg
                            in history):
                        logger.error(f"Invalid history format for user {user_id}, skipping")
                        continue
                    history_json = json.dumps(history)
                    c.execute("INSERT OR REPLACE INTO chat_history (user_id, history) VALUES (?, ?)",
                              (user_id, history_json))
                except json.JSONDecodeError:
                    logger.error(f"Failed to serialize history for user {user_id}")
                    continue

            for user_id, status in AUTO_REPLY_STATUS.items():
                c.execute("INSERT OR REPLACE INTO auto_reply_status (user_id, disabled_by_keyword) VALUES (?, ?)",
                          (user_id, status["disabled_by_keyword"]))

            conn.commit()

        logger.info(
            f"Data saved: {len(auto_reply_users)} auto-reply users, {len(CHAT_HISTORY)} chat history users, {len(AUTO_REPLY_STATUS)} auto-reply status entries")
    except Exception as e:
        logger.error(f"Error saving data: {e}")


async def get_dialog_user_list(client: TelegramClient) -> List[Dict]:
    """
    Retrieves a list of users from Telegram dialogs.

    Args:
        client: Initialized TelegramClient instance.

    Returns:
        List of dictionaries containing user information (id, name, auto_reply status, disabled_by_keyword).
    """
    try:
        users = []
        async for dialog in client.iter_dialogs():
            entity = dialog.entity
            if isinstance(entity, User):  # Only include user dialogs (not groups/channels)
                user_data = {
                    "id": entity.id,
                    "name": entity.first_name or entity.username or "Unknown",
                    "auto_reply": entity.id in auto_reply_users,
                    "disabled_by_keyword": AUTO_REPLY_STATUS.get(entity.id, {}).get("disabled_by_keyword", None)
                }
                users.append(user_data)
        logger.info(f"Retrieved {len(users)} users from dialogs")
        return users
    except Exception as e:
        logger.error(f"Error retrieving dialog user list: {e}")
        return []


async def get_last_messages(client: TelegramClient, user_ids: List[int]) -> Dict[int, List[Dict]]:
    """
    Retrieves the last 10 messages for specified user IDs.

    Args:
        client: Initialized TelegramClient instance.
        user_ids: List of user IDs to fetch messages for.

    Returns:
        Dictionary mapping user IDs to lists of their last 10 messages.
    """
    try:
        messages = {}
        for user_id in user_ids:
            user_messages = []
            async for message in client.iter_messages(user_id, limit=10):
                sender = await message.get_sender()
                user_messages.append({
                    "sender": sender.first_name or sender.username or "Unknown" if sender else "Unknown",
                    "timestamp": message.date.strftime("%Y-%m-%d %H:%M:%S"),
                    "text": message.text or ""
                })
            messages[user_id] = user_messages[::-1]  # Reverse to show newest first
        logger.info(f"Retrieved messages for {len(messages)} users")
        return messages
    except Exception as e:
        logger.error(f"Error retrieving messages: {e}")
        return {}