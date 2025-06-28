import asyncio
import signal
import os
from fastapi import FastAPI
from telethon import TelegramClient
from telethon.sessions import StringSession
from openai import AsyncOpenAI, OpenAIError
from dotenv import load_dotenv
import uvicorn
from utils import load_config, validate_env_vars, save_session, logger
from web_interface import init_web_routes
from chatgpt_handler import init_telegram_handlers

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables")
    raise OpenAIError("OPENAI_API_KEY must be set in the .env file or environment variables")

# Check templates directory
if not os.path.exists("templates"):
    logger.error("Templates directory not found")
    raise FileNotFoundError("Templates directory not found")

# Initialize FastAPI app
app = FastAPI()
logger.info("FastAPI app initialized")

# Initialize Telegram client
session_string = None
if os.path.exists("web_session.session"):
    try:
        with open("web_session.session", "r") as f:
            session_string = f.read().strip()
        logger.info("Loaded session from web_session.session")
    except Exception as e:
        logger.error(f"Failed to load session file: {e}")

client = TelegramClient(
    StringSession(session_string),
    int(os.getenv("TG_API_ID")),
    os.getenv("TG_API_HASH")
)
logger.info("Telegram client initialized")

# Initialize OpenAI client
client_gpt = AsyncOpenAI(api_key=OPENAI_API_KEY)
logger.info(f"Initialized client_gpt with API key: {OPENAI_API_KEY[:4]}...")

# Global state
started = False
app.state.client = client
app.state.client_gpt = client_gpt

async def initialize_telegram_client():
    logger.info("Initializing Telegram client")
    try:
        await client.connect()
        if not await client.is_user_authorized():
            logger.info("User not authorized, requesting Telegram login code")
            await client.send_code_request(os.getenv("TG_PHONE"))
            code = input("Enter Telegram login code: ")
            await client.sign_in(os.getenv("TG_PHONE"), code)
            save_session(client)
        logger.info("Telegram client connected and authorized")
        # Test connection with a simple API call
        me = await client.get_me()
        logger.info(f"Authorized as: {me.username or me.first_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram client: {e}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    global started
    if not started:
        logger.info("Starting application initialization")
        try:
            logger.info("Validating environment variables")
            validate_env_vars()
            logger.info("Environment variables validated")
            logger.info("Initializing Telegram client")
            await initialize_telegram_client()
            logger.info("Telegram client initialization completed")
            logger.info("Before initializing web routes")
            try:
                init_web_routes(app, client, client_gpt)
                logger.info("After initializing web routes")
            except Exception as e:
                logger.error(f"Error during web routes initialization: {e}")
                raise
            logger.info("Initializing Telegram handlers")
            init_telegram_handlers(client, client_gpt)
            logger.info("Telegram handlers initialized")
            # Run Telegram client in the background with error handling
            async def run_telegram_client():
                try:
                    await client.run_until_disconnected()
                except Exception as e:
                    logger.error(f"Telegram client disconnected with error: {e}")
            client.loop.create_task(run_telegram_client())
            logger.info("Telegram client running in background")
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
        started = True
        logger.info("Application startup completed")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    try:
        save_session(client)
        await client.disconnect()
        logger.info("Telegram client disconnected")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

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
        logger.info("Starting Uvicorn server")
        uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio", reload=True)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
    finally:
        loop.run_until_complete(handle_shutdown())