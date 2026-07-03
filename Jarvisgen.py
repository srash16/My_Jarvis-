from dotenv import load_dotenv
load_dotenv()
from google import genai
from elevenlabs.client import ElevenLabs
import edge_tts
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import sqlite3
import re
import asyncio
from datetime import datetime
import time


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ElevenLabs API key (free tier only supports certain premade voices via API)
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
ELEVENLABS_VOICE_IDS = [
    v for v in [
        os.getenv("ELEVENLABS_VOICE_ID"),
        "pNInz6obpgDQGcFmaJgB",  # Adam
        "EXAVITQu4vr4xnSDxMaL",  # Bella
    ] if v
]
EDGE_TTS_VOICE = "en-GB-RyanNeural"
print("Loading Whisper..... (first run downloads 140 MB, one time only )")
whisper_model = whisper.load_model("base")
print("JARVIS is online..... Type 'quit' to exit\n")

SAMPLE_RATE = 16000
DURATION=5 #SECONDS

# Database setup for memory persistence
DB_PATH = "jarvis_memory.db"

def init_db():
    """Initialize the SQLite database for storing conversations"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_text TEXT NOT NULL,
            jarvis_response TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_conversation(user_text, jarvis_response):
    """Save a conversation exchange to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversations (timestamp, user_text, jarvis_response)
        VALUES (?, ?, ?)
    ''', (datetime.now().isoformat(), user_text, jarvis_response))
    conn.commit()
    conn.close()

def get_recent_conversations(limit=10):
    """Retrieve recent conversations for context"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_text, jarvis_response FROM conversations
        ORDER BY timestamp DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    # Return in chronological order (oldest first)
    return [{'user': row[0], 'jarvis': row[1]} for row in reversed(rows)]

def record_audio():
    print("🎤 Listening...")
    audio=sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.flatten()

def transcribe(audio_array):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(f.name, audio_array, SAMPLE_RATE)
    try:
        result = whisper_model.transcribe(temp_path)
        return result["text"].strip()
    finally:
        os.unlink(temp_path)

def make_content(role, text):
    """Format a message for the google-genai SDK (role + parts, not content)."""
    return {"role": role, "parts": [{"text": text}]}

def strip_markdown(text):
    """Remove markdown so TTS reads naturally."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", " ", text)
    return text.strip()

def play_audio_bytes(audio_bytes):
    """Play MP3 bytes through the default output device."""
    import io
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    sd.play(data, samplerate)
    sd.wait()

def speak_elevenlabs(text):
    for voice_id in ELEVENLABS_VOICE_IDS:
        try:
            audio = b"".join(elevenlabs.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_flash_v2_5",
            ))
            if audio:
                play_audio_bytes(audio)
                return True
        except Exception:
            continue
    return False

async def _edge_tts_generate(text):
    audio = b""
    async for chunk in edge_tts.Communicate(text, EDGE_TTS_VOICE).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio

def speak_edge_tts(text):
    audio = asyncio.run(_edge_tts_generate(text))
    if audio:
        play_audio_bytes(audio)
        return True
    return False

def speak(text):
    print(f"JARVIS: {text}\n")
    spoken = strip_markdown(text)
    if not spoken:
        return
    if speak_elevenlabs(spoken):
        return
    print("(ElevenLabs unavailable, using free voice fallback...)")
    if speak_edge_tts(spoken):
        return
    print("Could not play audio.")

# Initialize database on startup
init_db()

# Load recent conversations to build initial conversation history
recent_chats = get_recent_conversations(5)
conversation_history = []
for chat in recent_chats:
    conversation_history.append(make_content("user", chat['user']))
    conversation_history.append(make_content("model", chat['jarvis']))

while True:
    input("Press Enter to speak...")  # just a trigger, no typing
    # Voice only
    audio = record_audio()
    text = transcribe(audio)

    if not text:
        print("(Nothing heard, try again)\n")
        continue

    if "goodbye" in text.lower() or "quit" in text.lower():
        speak("Going offline. Goodbye.")
        break

    print(f"You said: {text}")

    # Add user message to history
    conversation_history.append(make_content("user", text))

    # Get Gemini response with full conversation history
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversation_history
    )
    reply = response.text

    # Add model response to history
    conversation_history.append(make_content("model", reply))

    # Save conversation to memory
    save_conversation(text, reply)

    speak(reply)

    # Delay to avoid hitting rate limits
    time.sleep(1)