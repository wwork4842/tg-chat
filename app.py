import os
import mimetypes
from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from telethon import TelegramClient
from telethon.tl.types import User
from telethon.events import NewMessage
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
PHONE_NUMBER = os.getenv("TG_PHONE")

client_gpt = AsyncOpenAI(api_key=OPENAI_API_KEY)
client = TelegramClient("web_session", API_ID, API_HASH)
app = FastAPI()
templates = Jinja2Templates(directory="templates")
MEDIA_FOLDER = "media"
started = False

# ChatGPT Response Helper
async def ask_chatgpt(prompt: str) -> str:
    try:
        response = await client_gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error from OpenAI: {e}")
        return "Sorry, I couldn't generate a reply right now."

# Incoming message handler
@client.on(NewMessage(incoming=True))
async def handle_message(event):
    sender = await event.get_sender()
    text = event.raw_text
    if not text.strip():
        return
    response = await ask_chatgpt(text)
    await client.send_message(sender.id, response)

# Startup
@app.on_event("startup")
async def startup_event():
    global started
    os.makedirs(MEDIA_FOLDER, exist_ok=True)
    if not started:
        await client.connect()
        if not await client.is_user_authorized():
            await client.send_code_request(PHONE_NUMBER)
            code = input("Enter Telegram login code: ")
            await client.sign_in(PHONE_NUMBER, code)
        started = True

# Helper: List dialogs
async def get_dialogs():
    return [(d.id, d.name) for d in await client.get_dialogs()]

# Helper: List last messages
async def get_last_messages(chat_id: int, limit: int = 10):
    messages = []
    async for msg in client.iter_messages(chat_id, limit=limit):
        if msg.text:
            messages.append({"type": "text", "content": f"{msg.sender_id}: {msg.text}"})
        elif msg.media:
            filename = f"{msg.id}_{msg.file.name or 'file'}"
            filepath = os.path.join(MEDIA_FOLDER, filename)
            if not os.path.exists(filepath):
                await msg.download_media(file=filepath)
            ext = os.path.splitext(filename)[1].lower()
            media_type = (
                "image" if ext in [".jpg", ".jpeg", ".png", ".gif"]
                else "video" if ext in [".mp4", ".mov", ".webm"]
                else "audio" if ext in [".mp3", ".wav", ".ogg"]
                else "file"
            )
            messages.append({"type": media_type, "filename": filename, "sender": msg.sender_id})
    return messages[::-1]

# Helper: Users only
async def get_dialog_user_list():
    users = []
    async for dialog in client.iter_dialogs():
        entity = dialog.entity
        if isinstance(entity, User) and not entity.bot:
            name = f"{dialog.name} (@{entity.username})" if entity.username else dialog.name
            users.append({"id": dialog.id, "name": name})
    return users

# Main chat UI
@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    dialogs = await get_dialogs()
    return templates.TemplateResponse("index.html", {"request": request, "dialogs": dialogs, "messages": []})

@app.post("/", response_class=HTMLResponse)
async def send_message(request: Request, chat_id: int = Form(...), message: str = Form(None), read: str = Form(None)):
    dialogs = await get_dialogs()
    messages = []
    if message:
        await client.send_message(chat_id, message)
    if read:
        messages = await get_last_messages(chat_id)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "dialogs": dialogs,
        "messages": messages,
        "active_chat": chat_id,
        "success": bool(message)
    })

# Upload media
@app.post("/upload")
async def upload_file(chat_id: int = Form(...), file: UploadFile = File(...)):
    file_path = os.path.join(MEDIA_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    await client.send_file(chat_id, file_path)
    return JSONResponse(content={"success": True})

# List media
@app.get("/files", response_class=HTMLResponse)
async def list_files(request: Request):
    files = []
    for filename in os.listdir(MEDIA_FOLDER):
        path = os.path.join(MEDIA_FOLDER, filename)
        if os.path.isfile(path):
            ext = os.path.splitext(filename)[1].lower()
            media_type = (
                "image" if ext in [".jpg", ".jpeg", ".png", ".gif"]
                else "video" if ext in [".mp4", ".mov", ".webm"]
                else "audio" if ext in [".mp3", ".wav", ".ogg"]
                else "file"
            )
            files.append({"name": filename, "type": media_type})
    return templates.TemplateResponse("files.html", {"request": request, "files": files})

# Serve media
@app.get("/media/{filename}")
async def get_media(filename: str):
    path = os.path.join(MEDIA_FOLDER, filename)
    if os.path.exists(path):
        mime_type, _ = mimetypes.guess_type(path)
        return FileResponse(path, media_type=mime_type or "application/octet-stream")
    return JSONResponse(content={"error": "File not found"}, status_code=404)

# GPT-to-user message interface
@app.get("/send-gpt-message", response_class=HTMLResponse)
async def gpt_message_form(request: Request):
    users = await get_dialog_user_list()
    return templates.TemplateResponse("send_gpt_message.html", {"request": request, "users": users})

@app.post("/send-gpt-message", response_class=HTMLResponse)
async def send_gpt_message(request: Request, user_id: int = Form(...), instruction: str = Form(...)):
    try:
        generated_message = await ask_chatgpt(
            f"Write a polite and clear message to the user based on this instruction: {instruction}"
        )
        await client.send_message(user_id, generated_message)
        return templates.TemplateResponse("send_gpt_message.html", {
            "request": request,
            "users": await get_dialog_user_list(),
            "success": True,
            "sent_text": generated_message
        })
    except Exception as e:
        return templates.TemplateResponse("send_gpt_message.html", {
            "request": request,
            "users": await get_dialog_user_list(),
            "error": str(e)
        })
