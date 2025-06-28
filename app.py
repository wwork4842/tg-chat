import asyncio
import os
import logging
import json
import time
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import SessionPasswordNeededError, PhoneCodeInvalidError
from openai import AsyncOpenAI, OpenAIError
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import aiofiles
from pydantic import BaseModel

from utils import (
    load_config, validate_env_vars, save_session, load_data,
    auto_reply_users, AUTO_REPLY_DISABLE_KEYWORDS, NOTIFICATION_USER_ID
)
from web_interface import init_web_routes
from chatgpt_handler import init_telegram_handlers

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TelegramAuthRequest(BaseModel):
    phone_code: str
    password: Optional[str] = None


class ConfigManager:
    """Менеджер конфігурації з підтримкою гарячого перезавантаження"""

    def __init__(self):
        self._config = None
        self._last_modified = 0
        self._env_file = '.env'

    def get_config(self):
        if self._needs_reload():
            self._reload_config()
        return self._config

    def _needs_reload(self):
        try:
            if not os.path.exists(self._env_file):
                return False
            current_time = os.path.getmtime(self._env_file)
            return current_time > self._last_modified
        except Exception as e:
            logger.warning(f"Error checking config file modification time: {e}")
            return False

    def _reload_config(self):
        try:
            load_dotenv(override=True)
            self._last_modified = os.path.getmtime(self._env_file)
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")


class SecureSessionManager:
    """Безпечний менеджер сесій Telegram"""

    def __init__(self):
        self.session_file = "web_session.enc"
        self.key_env = "SESSION_ENCRYPTION_KEY"
        self._ensure_encryption_key()

    def _ensure_encryption_key(self):
        """Створює ключ шифрування якщо його немає"""
        if not os.getenv(self.key_env):
            key = Fernet.generate_key()
            logger.warning(f"Generated new encryption key. Add to .env: {self.key_env}={key.decode()}")
            os.environ[self.key_env] = key.decode()

    def _get_cipher(self) -> Fernet:
        key = os.getenv(self.key_env)
        if not key:
            raise ValueError("Encryption key not found")
        return Fernet(key.encode())

    async def load_session(self) -> Optional[str]:
        """Завантажує зашифровану сесію"""
        try:
            if not os.path.exists(self.session_file):
                return None

            cipher = self._get_cipher()
            async with aiofiles.open(self.session_file, 'rb') as f:
                encrypted_data = await f.read()

            decrypted_data = cipher.decrypt(encrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Failed to load encrypted session: {e}")
            return None

    async def save_session(self, session_string: str):
        """Зберігає зашифровану сесію"""
        try:
            cipher = self._get_cipher()
            encrypted_data = cipher.encrypt(session_string.encode())

            async with aiofiles.open(self.session_file, 'wb') as f:
                await f.write(encrypted_data)

            logger.info("Session saved successfully")
        except Exception as e:
            logger.error(f"Failed to save encrypted session: {e}")
            raise


class TelegramClientManager:
    """Менеджер Telegram клієнта з покращеною обробкою помилок"""

    def __init__(self, session_manager: SecureSessionManager):
        self.session_manager = session_manager
        self.client: Optional[TelegramClient] = None
        self.auth_pending = False
        self.phone_code_hash = None
        self.phone_number = None
        self.max_retries = 3
        self.retry_delay = 5

    async def initialize(self) -> TelegramClient:
        """Ініціалізує Telegram клієнт"""
        session_string = await self.session_manager.load_session()

        self.client = TelegramClient(
            StringSession(session_string),
            int(os.getenv("TG_API_ID")),
            os.getenv("TG_API_HASH")
        )

        await self._connect_with_retry()
        return self.client

    async def _connect_with_retry(self):
        """Підключення з повторними спробами"""
        for attempt in range(self.max_retries):
            try:
                await self.client.connect()
                logger.info("Telegram client connected successfully")
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

    async def is_authorized(self) -> bool:
        """Перевіряє чи авторизований клієнт"""
        try:
            return await self.client.is_user_authorized()
        except Exception as e:
            logger.error(f"Error checking authorization: {e}")
            return False

    async def start_auth(self, phone_number: str) -> str:
        """Починає процес авторизації"""
        try:
            self.phone_number = phone_number
            sent_code = await self.client.send_code_request(phone_number)
            self.phone_code_hash = sent_code.phone_code_hash
            self.auth_pending = True
            return "Code sent successfully"
        except Exception as e:
            logger.error(f"Failed to send code: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to send code: {str(e)}")

    async def complete_auth(self, phone_code: str, password: Optional[str] = None) -> str:
        """Завершує авторизацію"""
        if not self.auth_pending or not self.phone_code_hash:
            raise HTTPException(status_code=400, detail="No authentication in progress")

        try:
            await self.client.sign_in(
                self.phone_number,
                phone_code,
                phone_code_hash=self.phone_code_hash
            )
        except SessionPasswordNeededError:
            if not password:
                raise HTTPException(status_code=400, detail="Two-factor authentication password required")
            await self.client.sign_in(password=password)
        except PhoneCodeInvalidError:
            raise HTTPException(status_code=400, detail="Invalid phone code")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

        # Зберігаємо сесію
        session_string = self.client.session.save()
        await self.session_manager.save_session(session_string)

        self.auth_pending = False
        self.phone_code_hash = None
        self.phone_number = None

        return "Authentication completed successfully"

    async def disconnect(self):
        """Відключає клієнт"""
        if self.client and self.client.is_connected():
            session_string = self.client.session.save()
            await self.session_manager.save_session(session_string)
            await self.client.disconnect()


class HealthChecker:
    """Моніторинг здоров'я додатку"""

    def __init__(self, tg_manager: TelegramClientManager, gpt_client: AsyncOpenAI):
        self.tg_manager = tg_manager
        self.gpt_client = gpt_client
        self.start_time = time.time()

    async def get_health_status(self) -> dict:
        """Повертає статус здоров'я системи"""
        status = {
            "status": "healthy",
            "uptime": time.time() - self.start_time,
            "timestamp": time.time(),
            "services": {}
        }

        # Перевіряємо Telegram
        try:
            if self.tg_manager.client and await self.tg_manager.is_authorized():
                status["services"]["telegram"] = {"status": "connected", "authorized": True}
            else:
                status["services"]["telegram"] = {"status": "disconnected", "authorized": False}
                status["status"] = "degraded"
        except Exception as e:
            status["services"]["telegram"] = {"status": "error", "error": str(e)}
            status["status"] = "unhealthy"

        # Перевіряємо OpenAI
        try:
            # Простий тест API
            response = await self.gpt_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            status["services"]["openai"] = {"status": "connected"}
        except Exception as e:
            status["services"]["openai"] = {"status": "error", "error": str(e)}
            status["status"] = "unhealthy"

        return status


# Глобальні змінні
config_manager = ConfigManager()
session_manager = SecureSessionManager()
tg_manager = TelegramClientManager(session_manager)
health_checker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle менеджер для додатку"""
    global health_checker

    logger.info("Starting application...")

    try:
        # Валідація середовища
        validate_env_vars()

        # Ініціалізація OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise OpenAIError("OPENAI_API_KEY must be set")

        gpt_client = AsyncOpenAI(api_key=openai_key)

        # Ініціалізація Telegram
        tg_client = await tg_manager.initialize()

        # Зберігаємо клієнти в app state
        app.state.tg_client = tg_client
        app.state.gpt_client = gpt_client
        app.state.tg_manager = tg_manager
        app.state.config_manager = config_manager

        # Ініціалізуємо health checker
        health_checker = HealthChecker(tg_manager, gpt_client)
        app.state.health_checker = health_checker

        # Завантажуємо дані
        load_data()

        # Ініціалізуємо маршрути та обробники
        init_web_routes(app, tg_client, gpt_client)

        if await tg_manager.is_authorized():
            init_telegram_handlers(tg_client, gpt_client)
            logger.info("Telegram handlers initialized")

            # Запускаємо Telegram клієнт в фоні
            asyncio.create_task(tg_client.run_until_disconnected())
        else:
            logger.warning("Telegram not authorized. Please authenticate via /auth endpoints")

        logger.info("Application started successfully")
        yield

    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Shutting down application...")
        await tg_manager.disconnect()


def create_app() -> FastAPI:
    """Створює FastAPI додаток"""

    # Завантажуємо змінні середовища
    load_dotenv()

    app = FastAPI(
        title="Telegram ChatGPT Bot",
        description="Advanced Telegram bot with ChatGPT integration",
        version="2.0.0",
        lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # В production обмежте це
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        if health_checker:
            status = await health_checker.get_health_status()
            status_code = 200 if status["status"] == "healthy" else 503
            return JSONResponse(content=status, status_code=status_code)
        return {"status": "starting"}

    # Authentication endpoints
    @app.post("/auth/send-code")
    async def send_auth_code(phone_number: str):
        """Відправляє код авторизації"""
        try:
            result = await tg_manager.start_auth(phone_number)
            return {"message": result}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error sending auth code: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/auth/verify-code")
    async def verify_auth_code(auth_request: TelegramAuthRequest):
        """Перевіряє код авторизації"""
        try:
            result = await tg_manager.complete_auth(
                auth_request.phone_code,
                auth_request.password
            )

            # Після успішної авторизації ініціалізуємо обробники
            if await tg_manager.is_authorized():
                init_telegram_handlers(app.state.tg_client, app.state.gpt_client)
                asyncio.create_task(app.state.tg_client.run_until_disconnected())

            return {"message": result}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error verifying auth code: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/auth/status")
    async def auth_status():
        """Перевіряє статус авторизації"""
        try:
            is_authorized = await tg_manager.is_authorized()
            return {
                "authorized": is_authorized,
                "auth_pending": tg_manager.auth_pending
            }
        except Exception as e:
            logger.error(f"Error checking auth status: {e}")
            return {"authorized": False, "auth_pending": False}

    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    return app


# Створюємо додаток
app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Налаштування для різних середовищ
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug_mode,
        log_level="info" if not debug_mode else "debug"
    )