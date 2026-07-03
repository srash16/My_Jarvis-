from dotenv import load_dotenv
load_dotenv()

import asyncio
import io
import os
import re
import sqlite3
import tempfile
import threading
import time
from datetime import datetime

import edge_tts
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from elevenlabs.client import ElevenLabs
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

ELEVENLABS_VOICE_IDS = [
    v for v in [
        os.getenv("ELEVENLABS_VOICE_ID"),
        "pNInz6obpgDQGcFmaJgB",  # Adam (free tier)
        "EXAVITQu4vr4xnSDxMaL",  # Bella (free tier)
    ] if v
]
EDGE_TTS_VOICE = "en-GB-RyanNeural"

SAMPLE_RATE = 16000
COMMAND_DURATION = 5  # seconds to record after wake word
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
ENERGY_THRESHOLD = 0.01
SILENCE_THRESHOLD = 0.5
WAKE_WORD = "hey jarvis"
EXIT_WORDS = ("goodbye", "quit", "exit", "bye", "stop")
DB_PATH = "jarvis_memory.db"

stop_listening = False
is_processing = False
conversation_history = []

print("Loading Whisper..... (first run downloads 140 MB, one time only )")
whisper_model = whisper.load_model("base")
print("JARVIS is online..... Say 'Hey Jarvis' to activate\n")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_text TEXT NOT NULL,
            jarvis_response TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_conversation(user_text, jarvis_response):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO conversations (timestamp, user_text, jarvis_response) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), user_text, jarvis_response),
    )
    conn.commit()
    conn.close()


def get_recent_conversations(limit=10):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT user_text, jarvis_response FROM conversations ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"user": row[0], "jarvis": row[1]} for row in reversed(rows)]


def transcribe(audio_array):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio_array, SAMPLE_RATE)
    try:
        return whisper_model.transcribe(temp_path)["text"].strip()
    finally:
        os.unlink(temp_path)


def record_audio(duration=COMMAND_DURATION):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def make_content(role, text):
    return {"role": role, "parts": [{"text": text}]}


def strip_markdown(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\n+", " ", text).strip()


def play_audio_bytes(audio_bytes):
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
    speak_edge_tts(spoken)


def beep(frequency=1500, duration=0.1):
    def _play():
        fs = 44100
        t = np.linspace(0, duration, int(fs * duration), False)
        note = np.sin(frequency * 2 * np.pi * t) * 0.3
        sd.play(note.astype(np.float32), fs)
        sd.wait()

    threading.Thread(target=_play, daemon=True).start()


def voice_detected(chunk):
    return np.sqrt(np.mean(np.square(chunk))) > ENERGY_THRESHOLD


def extract_command(transcript):
    """Return command text after wake word, or None if wake word not present."""
    text = transcript.lower()
    if WAKE_WORD not in text:
        return None
    return text.replace(WAKE_WORD, "", 1).strip()


def listen_for_command():
    text = transcribe(record_audio())
    if text:
        process_command(text)


def process_command(text):
    global stop_listening, conversation_history

    print(f"You said: {text}")

    if any(word in text.lower() for word in EXIT_WORDS):
        speak("Going offline. Goodbye.")
        stop_listening = True
        return

    conversation_history.append(make_content("user", text))

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=conversation_history,
        )
        reply = response.text
        conversation_history.append(make_content("model", reply))
        save_conversation(text, reply)
        speak(reply)
    except Exception as e:
        print(f"Error getting response: {e}")
        speak("I'm having trouble thinking right now. Please try again.")

    time.sleep(0.5)


def listen_for_wake_word():
    """Continuously listen for 'Hey Jarvis' using voice-activity chunks."""
    global is_processing, stop_listening

    print("👂 Listening for wake word...")
    audio_buffer = []
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_THRESHOLD / CHUNK_DURATION)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while not stop_listening:
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk = chunk.flatten()

            if voice_detected(chunk):
                audio_buffer.extend(chunk)
                silent_chunks = 0
                continue

            if not audio_buffer:
                continue

            silent_chunks += 1
            if silent_chunks <= max_silent_chunks:
                continue

            audio_data = np.array(audio_buffer, dtype=np.float32)
            audio_buffer = []
            silent_chunks = 0

            if is_processing or stop_listening:
                continue

            transcript = transcribe(audio_data)
            command = extract_command(transcript)
            if command is None:
                continue

            print("🎯 Wake word detected!")
            beep()
            is_processing = True
            try:
                if command:
                    process_command(command)
                else:
                    listen_for_command()
            finally:
                is_processing = False


def start_voice_interaction():
    global stop_listening, conversation_history

    init_db()
    recent_chats = get_recent_conversations(5)
    conversation_history = []
    for chat in recent_chats:
        conversation_history.append(make_content("user", chat["user"]))
        conversation_history.append(make_content("model", chat["jarvis"]))

    stop_listening = False

    listen_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    listen_thread.start()

    print("🗣️  Say 'Hey Jarvis' to activate, then speak your command")
    print("💡 Say 'goodbye', 'quit', or 'exit' to stop")

    try:
        while not stop_listening:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        stop_listening = True
        listen_thread.join(timeout=1.0)


if __name__ == "__main__":
    start_voice_interaction()
