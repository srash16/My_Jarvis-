from dotenv import load_dotenv
load_dotenv()
from google import genai
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

chat = client.chats.create(
    model="gemini-2.5-flash",
    config={
        "system_instruction": (
            "You are JARVIS, a helpful and intelligent AI assistant. "
            "Be concise, smart, and slightly witty — like Tony Stark's AI. "
            "Keep responses under 3 sentences unless asked for detail."
        )
    }
)

eleven= ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID=os.getenv("ELEVENLABS_VOICE_ID")
print("Loading Whisper..... (first run downloads 140 MB, one time only )")
whisper_model = whisper.load_model("base")
print("JARVIS is online..... Type 'quit' to exit\n")

SAMPLE_RATE = 16000
DURATION=5 #SECONDS

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

def speak(text):
    print(f"JARVIS: {text}\n")
    try:
        audio = eleven.generate(
            text=text,
            voice=VOICE_ID,
            model="eleven_monolingual_v1"
        )
        play(audio)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        # Fallback to just printing if TTS fails
        print(f"JARVIS (TTS failed): {text}")

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

    # Get Gemini response
    response = chat.send_message(text)
    reply = response.text
    speak(reply)