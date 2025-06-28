import asyncio
import os
import logging
from fastapi import FastAPI
from telethon import TelegramClient
from telethon.sessions import StringSession
from openai import AsyncOpenAI, OpenAIError
from dotenv import load_dotenv

from utils import (
    load_config, validate_env_vars, save_session, load_data,
    auto_reply_users, AUTO_REPLY_DISABLE_KEYWORDS, NOTIFICATION_USER_ID
)
from web_interface import init_web_routes
from chatgpt_handler import init_telegram_handlers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI()
    load_dotenv()

    # Validate environment
    try:
        validate_env_vars()
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        raise

    # Load OpenAI API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.critical("OPENAI_API_KEY is not set")
        raise OpenAIError("OPENAI_API_KEY must be set in the .env file or environment variables")

    # Load Telegram session
    session_string = None
    if os.path.exists("web_session.session"):
        with open("web_session.session", "r") as f:
            session_string = f.read().strip()

    # Initialize clients
    tg_client = TelegramClient(
        StringSession(session_string),
        int(os.getenv("TG_API_ID")),
        os.getenv("TG_API_HASH")
    )
    gpt_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Store in app state
    app.state.tg_client = tg_client
    app.state.gpt_client = gpt_client

    @app.on_event("startup")
    async def on_startup():
        logger.info("Loading data and initializing clients...")
        load_data()
        await tg_client.connect()
        if not await tg_client.is_user_authorized():
            logger.info("Telegram not authorized. Please log in.")
            await tg_client.send_code_request(os.getenv("TG_PHONE"))
            code = input("Enter Telegram login code: ")
            await tg_client.sign_in(os.getenv("TG_PHONE"), code)
            save_session(tg_client)
        logger.info("Telegram client authorized.")
        init_web_routes(app, tg_client, gpt_client)
        init_telegram_handlers(tg_client, gpt_client)
        logger.info("Web routes and Telegram handlers initialized.")
        asyncio.create_task(tg_client.run_until_disconnected())

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("Shutting down Telegram client...")
        save_session(tg_client)
        await tg_client.disconnect()

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)