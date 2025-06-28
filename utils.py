import os
import json
import sqlite3
import logging
from typing import List, Dict, Optional, Set, Any
from telethon import TelegramClient
from telethon.tl.types import User
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging explicitly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigKey(Enum):
    """Enum для ключів конфігурації"""
    AUTO_REPLY_FILE = "auto_reply_users.json"
    DB_PATH = "chat_history.db"
    SESSION_FILE = "web_session.session"
    KEYWORDS_FILE = "keywords.json"
    CONFIG_FILE = "config.json"
    SQLITE_TIMEOUT = 30


@dataclass
class UserData:
    """Клас для представлення даних користувача"""
    id: int
    name: str
    auto_reply: bool = False
    disabled_by_keyword: Optional[str] = None


@dataclass
class MessageData:
    """Клас для представлення повідомлення"""
    sender: str
    timestamp: str
    text: str


@dataclass
class ChatMessage:
    """Клас для повідомлення в історії чату"""
    role: str
    content: str
    timestamp: str


class ConfigManager:
    """Менеджер конфігурації з кешуванням та валідацією"""

    def __init__(self, config_file: str = ConfigKey.CONFIG_FILE.value):
        self.config_file = Path(config_file)
        self._cache: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Завантажує конфігурацію з файлу"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logger.info(f"Config loaded from {self.config_file}")
            else:
                logger.warning(f"Config file {self.config_file} not found, using defaults")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Отримує значення з конфігурації"""
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Встановлює значення в конфігурації"""
        self._cache[key] = value
        self._save_config()

    def _save_config(self) -> None:
        """Зберігає конфігурацію у файл"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Config saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")


class DatabaseManager:
    """Менеджер бази даних з покращеною обробкою помилок"""

    def __init__(self, db_path: str = ConfigKey.DB_PATH.value, timeout: int = ConfigKey.SQLITE_TIMEOUT.value):
        self.db_path = Path(db_path)
        self.timeout = timeout
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Ініціалізує базу даних з міграцією"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Створюємо таблиці, якщо вони не існують
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS chat_history
                               (
                                   user_id
                                   INTEGER
                                   PRIMARY
                                   KEY,
                                   history
                                   TEXT
                               )
                               ''')
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS auto_reply_status
                               (
                                   user_id
                                   INTEGER
                                   PRIMARY
                                   KEY,
                                   disabled_by_keyword
                                   TEXT
                               )
                               ''')

                # Перевіряємо і додаємо колонки updated_at, якщо їх немає
                self._migrate_database(cursor)

                # Додаємо індекси тільки якщо колонки існують
                try:
                    cursor.execute('SELECT updated_at FROM chat_history LIMIT 1')
                    cursor.execute('''
                                   CREATE INDEX IF NOT EXISTS idx_chat_history_updated
                                       ON chat_history(updated_at)
                                   ''')
                except sqlite3.OperationalError:
                    # Колонка updated_at не існує, пропускаємо створення індексу
                    pass

                conn.commit()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _migrate_database(self, cursor: sqlite3.Cursor) -> None:
        """Виконує міграцію бази даних"""
        try:
            # Перевіряємо чи існує колонка updated_at в chat_history
            cursor.execute("PRAGMA table_info(chat_history)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'updated_at' not in columns:
                logger.info("Adding updated_at column to chat_history table")
                cursor.execute('''
                               ALTER TABLE chat_history
                                   ADD COLUMN updated_at TIMESTAMP
                               ''')
                # Оновлюємо існуючі записи
                cursor.execute('''
                               UPDATE chat_history
                               SET updated_at = datetime('now')
                               WHERE updated_at IS NULL
                               ''')

            # Перевіряємо чи існує колонка updated_at в auto_reply_status
            cursor.execute("PRAGMA table_info(auto_reply_status)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'updated_at' not in columns:
                logger.info("Adding updated_at column to auto_reply_status table")
                cursor.execute('''
                               ALTER TABLE auto_reply_status
                                   ADD COLUMN updated_at TIMESTAMP
                               ''')
                # Оновлюємо існуючі записи
                cursor.execute('''
                               UPDATE auto_reply_status
                               SET updated_at = datetime('now')
                               WHERE updated_at IS NULL
                               ''')

            # Перевіряємо чи існує колонка history як NOT NULL
            cursor.execute("PRAGMA table_info(chat_history)")
            history_column = next((col for col in cursor.fetchall() if col[1] == 'history'), None)

            # Якщо history не має NOT NULL, оновлюємо NULL значення
            if history_column:
                cursor.execute('UPDATE chat_history SET history = "[]" WHERE history IS NULL')

        except Exception as e:
            logger.error(f"Error during database migration: {e}")
            # Не викидаємо помилку, щоб не зламати ініціалізацію

    @contextmanager
    def _get_connection(self):
        """Контекстний менеджер для підключення до БД"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def save_chat_history(self, user_id: int, history: List[ChatMessage]) -> bool:
        """Зберігає історію чату користувача"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                history_json = json.dumps([asdict(msg) for msg in history], ensure_ascii=False)

                # Перевіряємо чи існує колонка updated_at
                cursor.execute("PRAGMA table_info(chat_history)")
                columns = [column[1] for column in cursor.fetchall()]

                if 'updated_at' in columns:
                    cursor.execute('''
                        INSERT OR REPLACE INTO chat_history (user_id, history, updated_at) 
                        VALUES (?, ?, datetime('now'))
                    ''', (user_id, history_json))
                else:
                    cursor.execute('''
                        INSERT OR REPLACE INTO chat_history (user_id, history) 
                        VALUES (?, ?)
                    ''', (user_id, history_json))

                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving chat history for user {user_id}: {e}")
            return False

    def load_chat_history(self, user_id: int) -> Optional[List[ChatMessage]]:
        """Завантажує історію чату користувача"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT history FROM chat_history WHERE user_id = ?', (user_id,))
                row = cursor.fetchone()
                if row:
                    history_data = json.loads(row['history'])
                    return [ChatMessage(**msg) for msg in history_data]
            return None
        except Exception as e:
            logger.error(f"Error loading chat history for user {user_id}: {e}")
            return None

    def save_auto_reply_status(self, user_id: int, disabled_by_keyword: Optional[str]) -> bool:
        """Зберігає статус автовідповіді"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Перевіряємо чи існує колонка updated_at
                cursor.execute("PRAGMA table_info(auto_reply_status)")
                columns = [column[1] for column in cursor.fetchall()]

                if 'updated_at' in columns:
                    cursor.execute('''
                        INSERT OR REPLACE INTO auto_reply_status (user_id, disabled_by_keyword, updated_at) 
                        VALUES (?, ?, datetime('now'))
                    ''', (user_id, disabled_by_keyword))
                else:
                    cursor.execute('''
                        INSERT OR REPLACE INTO auto_reply_status (user_id, disabled_by_keyword) 
                        VALUES (?, ?)
                    ''', (user_id, disabled_by_keyword))

                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving auto reply status for user {user_id}: {e}")
            return False

    def load_all_auto_reply_status(self) -> Dict[int, Dict[str, Optional[str]]]:
        """Завантажує всі статуси автовідповіді"""
        try:
            result = {}
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT user_id, disabled_by_keyword FROM auto_reply_status')
                for row in cursor.fetchall():
                    if row['disabled_by_keyword'] is not None:
                        result[row['user_id']] = {"disabled_by_keyword": row['disabled_by_keyword']}
            return result
        except Exception as e:
            logger.error(f"Error loading auto reply status: {e}")
            return {}

    def reset_chat_history(self) -> bool:
        """Очищає всю історію чатів"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history")
                conn.commit()
            logger.info("Chat history reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting chat history: {e}")
            return False


class DataManager:
    """Менеджер для роботи з даними застосунку"""

    def __init__(self, config_manager: ConfigManager, db_manager: DatabaseManager):
        self.config = config_manager
        self.db = db_manager
        self.auto_reply_users: Set[int] = set()
        self.chat_history: Dict[int, List[ChatMessage]] = {}
        self.auto_reply_status: Dict[int, Dict[str, Optional[str]]] = {}
        self.auto_reply_disable_keywords: List[str] = []
        self.female_names = ["Аня", "Катя", "Маша", "Лена", "Настя", "Юля", "Оля", "Таня"]

    def load_all_data(self) -> None:
        """Завантажує всі дані"""
        try:
            self._load_auto_reply_users()
            self._load_keywords()
            self.auto_reply_status = self.db.load_all_auto_reply_status()
            logger.info("All data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def _load_auto_reply_users(self) -> None:
        """Завантажує список користувачів для автовідповіді"""
        try:
            auto_reply_file = Path(self.config.get("auto_reply_file", ConfigKey.AUTO_REPLY_FILE.value))
            if auto_reply_file.exists():
                with open(auto_reply_file, 'r', encoding='utf-8') as f:
                    users = json.load(f)
                    self.auto_reply_users.update(set(users))
                logger.info(f"Loaded {len(self.auto_reply_users)} auto-reply users")
        except Exception as e:
            logger.error(f"Error loading auto-reply users: {e}")

    def _load_keywords(self) -> None:
        """Завантажує ключові слова для відключення автовідповіді"""
        try:
            keywords_file = Path(self.config.get("keywords_file", ConfigKey.KEYWORDS_FILE.value))
            if keywords_file.exists():
                with open(keywords_file, 'r', encoding='utf-8') as f:
                    keywords = json.load(f)
                    if isinstance(keywords, list):
                        self.auto_reply_disable_keywords = [kw.strip().lower() for kw in keywords]
            else:
                self.auto_reply_disable_keywords = ["stop", "disable", "off", "стоп", "відключити"]
            logger.info(f"Loaded keywords: {self.auto_reply_disable_keywords}")
        except Exception as e:
            logger.error(f"Error loading keywords: {e}")
            self.auto_reply_disable_keywords = ["stop", "disable", "off", "стоп", "відключити"]

    def save_auto_reply_users(self) -> bool:
        """Зберігає список користувачів для автовідповіді"""
        try:
            auto_reply_file = Path(self.config.get("auto_reply_file", ConfigKey.AUTO_REPLY_FILE.value))
            auto_reply_file.parent.mkdir(parents=True, exist_ok=True)
            with open(auto_reply_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.auto_reply_users), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.auto_reply_users)} auto-reply users")
            return True
        except Exception as e:
            logger.error(f"Error saving auto-reply users: {e}")
            return False

    def save_keywords(self, keywords: List[str]) -> bool:
        """Зберігає ключові слова"""
        try:
            keywords_file = Path(self.config.get("keywords_file", ConfigKey.KEYWORDS_FILE.value))
            keywords_file.parent.mkdir(parents=True, exist_ok=True)
            with open(keywords_file, 'w', encoding='utf-8') as f:
                json.dump(keywords, f, indent=2, ensure_ascii=False)
            self.auto_reply_disable_keywords = [kw.strip().lower() for kw in keywords]
            logger.info(f"Keywords saved: {keywords}")
            return True
        except Exception as e:
            logger.error(f"Error saving keywords: {e}")
            return False

    def get_chat_history(self, user_id: int) -> List[ChatMessage]:
        """Отримує історію чату користувача"""
        if user_id not in self.chat_history:
            history = self.db.load_chat_history(user_id)
            self.chat_history[user_id] = history or []
        return self.chat_history[user_id]

    def add_message_to_history(self, user_id: int, message: ChatMessage) -> None:
        """Додає повідомлення до історії чату"""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        self.chat_history[user_id].append(message)
        # Обмежуємо історію останніми 100 повідомленнями
        if len(self.chat_history[user_id]) > 100:
            self.chat_history[user_id] = self.chat_history[user_id][-100:]
        self.db.save_chat_history(user_id, self.chat_history[user_id])


class TelegramUtils:
    """Утиліти для роботи з Telegram"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    async def get_dialog_user_list(self, client: TelegramClient) -> List[UserData]:
        """Отримує список користувачів з діалогів"""
        try:
            users = []
            async for dialog in client.iter_dialogs():
                entity = dialog.entity
                if isinstance(entity, User):
                    user_data = UserData(
                        id=entity.id,
                        name=entity.first_name or entity.username or "Unknown",
                        auto_reply=entity.id in self.data_manager.auto_reply_users,
                        disabled_by_keyword=self.data_manager.auto_reply_status.get(
                            entity.id, {}
                        ).get("disabled_by_keyword")
                    )
                    users.append(user_data)
            logger.info(f"Retrieved {len(users)} users from dialogs")
            return users
        except Exception as e:
            logger.error(f"Error retrieving dialog user list: {e}")
            return []

    async def get_last_messages(self, client: TelegramClient, user_ids: List[int], limit: int = 10) -> Dict[
        int, List[MessageData]]:
        """Отримує останні повідомлення для вказаних користувачів"""
        try:
            messages = {}
            for user_id in user_ids:
                user_messages = []
                async for message in client.iter_messages(user_id, limit=limit):
                    sender = await message.get_sender()
                    message_data = MessageData(
                        sender=sender.first_name or sender.username or "Unknown" if sender else "Unknown",
                        timestamp=message.date.strftime("%Y-%m-%d %H:%M:%S"),
                        text=message.text or ""
                    )
                    user_messages.append(message_data)
                messages[user_id] = list(reversed(user_messages))  # Найновіші спочатку
            logger.info(f"Retrieved messages for {len(messages)} users")
            return messages
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            return {}


def validate_environment() -> None:
    """Перевіряє наявність необхідних змінних середовища"""
    required_vars = ["OPENAI_API_KEY", "TG_API_ID", "TG_API_HASH", "TG_PHONE"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        error_msg = f"Missing environment variables: {', '.join(missing)}"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)


async def save_session_string(client: TelegramClient, session_file: str = ConfigKey.SESSION_FILE.value) -> bool:
    """Зберігає сесію Telegram"""
    try:
        session_path = Path(session_file)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_string = client.session.save()
        with open(session_path, 'w', encoding='utf-8') as f:
            f.write(session_string)
        logger.info(f"Session saved to {session_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        return False


# Фабрика для створення основних компонентів
def create_app_components() -> tuple[ConfigManager, DatabaseManager, DataManager, TelegramUtils]:
    """Створює та ініціалізує основні компоненти застосунку"""
    config_manager = ConfigManager()
    db_manager = DatabaseManager()
    data_manager = DataManager(config_manager, db_manager)
    telegram_utils = TelegramUtils(data_manager)

    # Завантажуємо дані
    data_manager.load_all_data()

    return config_manager, db_manager, data_manager, telegram_utils


# Для зворотної сумісності - глобальні змінні (deprecated)
config_manager, db_manager, data_manager, telegram_utils = create_app_components()

# Deprecated globals - використовуйте об'єкти вище
CHAT_HISTORY = data_manager.chat_history
AUTO_REPLY_STATUS = data_manager.auto_reply_status
auto_reply_users = data_manager.auto_reply_users
AUTO_REPLY_DISABLE_KEYWORDS = data_manager.auto_reply_disable_keywords
FEMALE_NAMES = data_manager.female_names
NOTIFICATION_USER_ID = config_manager.get("notification_user_id")

# CONFIG для зворотної сумісності
CONFIG = {
    "AUTO_REPLY_FILE": config_manager.get("auto_reply_file", ConfigKey.AUTO_REPLY_FILE.value),
    "DB_PATH": config_manager.get("db_path", ConfigKey.DB_PATH.value),
    "SESSION_FILE": config_manager.get("session_file", ConfigKey.SESSION_FILE.value),
    "KEYWORDS_FILE": config_manager.get("keywords_file", ConfigKey.KEYWORDS_FILE.value),
    "CONFIG_FILE": config_manager.get("config_file", ConfigKey.CONFIG_FILE.value),
    "SQLITE_TIMEOUT": config_manager.get("sqlite_timeout", ConfigKey.SQLITE_TIMEOUT.value),
}


# Deprecated functions - використовуйте методи класів
def load_config():
    return config_manager.get("notification_user_id")


def save_config(notification_user_id):
    config_manager.set("notification_user_id", notification_user_id)


def load_keywords():
    return data_manager.auto_reply_disable_keywords


def save_keywords(keywords):
    return data_manager.save_keywords(keywords)


def validate_env_vars():
    validate_environment()


def save_session(client):
    return asyncio.run(save_session_string(client))


def init_db():
    db_manager._initialize_db()


def reset_chat_history():
    return db_manager.reset_chat_history()


def load_data():
    data_manager.load_all_data()


def save_data():
    return data_manager.save_auto_reply_users()


async def get_dialog_user_list(client):
    return await telegram_utils.get_dialog_user_list(client)


async def get_last_messages(client, user_ids):
    return await telegram_utils.get_last_messages(client, user_ids)