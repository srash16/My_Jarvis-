from dotenv import load_dotenv
load_dotenv()

import asyncio
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import edge_tts
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from elevenlabs.client import ElevenLabs
from google import genai

from memory import JarvisMemory
from jarvis_brain import generate_with_tools, friendly_error
from local_commands import handle_locally
from system_control import SYSTEM_CONTROL_PROMPT
from system_config import GMAIL_ACCOUNTS, GMAIL_DEFAULT, GEMINI_API_KEY, VISION_ENABLED
from agent_setup import run_smart_agent
from vision import capture_and_describe_once

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

ELEVENLABS_VOICE_IDS = [
    v for v in [
        os.getenv("ELEVENLABS_VOICE_ID"),
        "pNInz6obpgDQGcFmaJgB",
        "EXAVITQu4vr4xnSDxMaL",
    ] if v
]
EDGE_TTS_VOICE = "en-GB-RyanNeural"
# Optional: force output device index or name substring, e.g. JARVIS_OUTPUT_DEVICE=Realtek
# If unset, prefers local Realtek speakers/headphones over TVs/HDMI.
OUTPUT_DEVICE_PREF = (os.getenv("JARVIS_OUTPUT_DEVICE") or "").strip()

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
MAX_COMMAND_SECONDS = 12
SILENCE_END_SECONDS = 1.2
MIN_SPEECH_SECONDS = 0.6
WAKE_SILENCE_SECONDS = 0.8
WHISPER_PROMPT = (
    "Jarvis, open chrome, notepad, calculator, folder, volume, brightness, "
    "desktop, downloads, documents, email, website."
)

stop_listening = False
is_processing = False
conversation_history = []
memory = None
energy_threshold = 0.008

print("Loading Whisper..... (first run downloads 140 MB, one time only )")
whisper_model = whisper.load_model("base")
print("JARVIS is online..... Say 'Jarvis' to activate\n")


def calibrate_mic(seconds=1.5):
    """Measure ambient noise and set voice detection threshold."""
    global energy_threshold
    print("🎚️  Calibrating mic (stay quiet)...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    noise = float(np.sqrt(np.mean(np.square(audio))))
    energy_threshold = max(0.004, noise * 2.5)
    print(f"🎚️  Mic calibrated (threshold: {energy_threshold:.4f})\n")


def normalize_audio(audio_array):
    peak = np.max(np.abs(audio_array))
    if peak > 0.001:
        return (audio_array / peak * 0.9).astype(np.float32)
    return audio_array


def transcribe(audio_array):
    audio_array = normalize_audio(np.asarray(audio_array, dtype=np.float32))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio_array, SAMPLE_RATE)
    try:
        result = whisper_model.transcribe(
            temp_path,
            language="en",
            fp16=False,
            initial_prompt=WHISPER_PROMPT,
        )
        return result["text"].strip()
    finally:
        os.unlink(temp_path)


def record_until_silence(max_seconds=MAX_COMMAND_SECONDS, silence_seconds=SILENCE_END_SECONDS):
    """Record from mic until user stops speaking or max time reached."""
    print("🎤 Listening... (speak now)")
    max_chunks = int(max_seconds / CHUNK_DURATION)
    silence_chunks = int(silence_seconds / CHUNK_DURATION)
    min_chunks = int(MIN_SPEECH_SECONDS / CHUNK_DURATION)

    buffer = []
    silent_run = 0
    heard_voice = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk = chunk.flatten()
            buffer.extend(chunk)

            if voice_detected(chunk):
                heard_voice = True
                silent_run = 0
            elif heard_voice:
                silent_run += 1
                if silent_run >= silence_chunks and len(buffer) >= min_chunks * CHUNK_SIZE:
                    break

    if not heard_voice:
        print("(No speech detected)")
        return np.array([], dtype=np.float32)
    return np.array(buffer, dtype=np.float32)


def make_content(role, text):
    return {"role": role, "parts": [{"text": text}]}


def strip_markdown(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\n+", " ", text).strip()


def _resolve_output_device():
    """Pick a local speaker/headphones device; avoid TVs/HDMI when possible."""
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    def name_of(i):
        try:
            return str(devices[i]["name"])
        except Exception:
            return ""

    # Explicit override from .env (index or name substring)
    if OUTPUT_DEVICE_PREF:
        if OUTPUT_DEVICE_PREF.isdigit():
            idx = int(OUTPUT_DEVICE_PREF)
            if 0 <= idx < len(devices) and devices[idx]["max_output_channels"] > 0:
                return idx
        needle = OUTPUT_DEVICE_PREF.lower()
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0 and needle in d["name"].lower():
                return i

    avoid = ("tv", "hdmi", "mapper", "primary sound", "nvidia", "amd high definition")
    prefer = ("realtek", "headphones", "headset", "speaker")

    candidates = [
        i for i, d in enumerate(devices)
        if d["max_output_channels"] > 0
        and not any(a in d["name"].lower() for a in avoid)
    ]
    for keyword in prefer:
        for i in candidates:
            if keyword in name_of(i).lower():
                return i
    if candidates:
        return candidates[0]

    # Last resort: PortAudio default output
    try:
        return sd.default.device[1]
    except Exception:
        return None


_OUTPUT_DEVICE = None  # resolved lazily once


def _get_output_device():
    global _OUTPUT_DEVICE
    if _OUTPUT_DEVICE is None:
        _OUTPUT_DEVICE = _resolve_output_device()
        try:
            name = sd.query_devices(_OUTPUT_DEVICE)["name"] if _OUTPUT_DEVICE is not None else "system default"
        except Exception:
            name = str(_OUTPUT_DEVICE)
        print(f"🔊 Audio output: {name}")
    return _OUTPUT_DEVICE


def play_audio_bytes(audio_bytes):
    """Play MP3/audio bytes through a local speaker (not TV/HDMI when avoidable)."""
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)

    device = _get_output_device()
    try:
        sd.play(data, samplerate, device=device, blocking=True)
        return
    except Exception as e:
        print(f"(sounddevice play failed: {e}; trying Windows fallback)")

    # Windows fallback — works even while the mic InputStream is open
    path = None
    try:
        import winsound
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        sf.write(path, data, samplerate)
        winsound.PlaySound(path, winsound.SND_FILENAME)
    except Exception as e:
        print(f"(audio fallback failed: {e})")
    finally:
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass


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
        except Exception as e:
            print(f"ElevenLabs TTS failed: {e}")
            continue
    return False


async def _edge_tts_generate(text):
    audio = b""
    async for chunk in edge_tts.Communicate(text, EDGE_TTS_VOICE).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio


def speak_edge_tts(text):
    try:
        audio = asyncio.run(_edge_tts_generate(text))
        if audio:
            play_audio_bytes(audio)
            return True
    except Exception as e:
        print(f"Edge TTS failed: {e}")
    return False


SPEAKING_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_speaking_state.json")
LISTENING_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_listening_state.json")


def _set_speaking_state(speaking):
    try:
        with open(SPEAKING_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"speaking": bool(speaking)}, f)
    except Exception:
        pass


def _set_listening_state(listening):
    try:
        with open(LISTENING_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"listening": bool(listening)}, f)
    except Exception:
        pass


def speak(text, already_printed=False):
    if not already_printed:
        print(f"JARVIS: {text}\n")
    spoken = strip_markdown(text)
    if not spoken:
        return
    _set_speaking_state(True)
    try:
        if speak_edge_tts(spoken):
            return
        print("(Edge TTS unavailable, trying ElevenLabs...)")
        if speak_elevenlabs(spoken):
            return
        print("(Could not play audio — check speakers.)")
    except Exception as e:
        print(f"TTS error: {e}")
    finally:
        _set_speaking_state(False)


def beep(frequency=1500, duration=0.15):
    def _play():
        try:
            import winsound
            winsound.Beep(int(frequency), int(duration * 1000))
            return
        except Exception:
            pass
        fs = 44100
        t = np.linspace(0, duration, int(fs * duration), False)
        note = np.sin(frequency * 2 * np.pi * t) * 0.3
        try:
            sd.play(note.astype(np.float32), fs, device=_get_output_device(), blocking=True)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


def voice_detected(chunk):
    return float(np.sqrt(np.mean(np.square(chunk)))) > energy_threshold


def extract_wake_word(transcript):
    """Return True if transcript contains wake word 'jarvis'."""
    return bool(re.search(r"\bjarvis\b", transcript.lower()))


def clean_command(text):
    text = text.lower().strip()
    text = re.sub(r"\bjarvis\b", "", text, count=1)
    return re.sub(r"^[,.\!\?\s]+", "", text).strip()


def listen_for_command():
    """Record a fresh command after wake word beep."""
    audio = record_until_silence()
    _set_listening_state(False)
    if len(audio) == 0:
        speak("I didn't catch that. Try again.")
        return
    text = transcribe(audio)
    if not text:
        speak("I didn't catch that. Try again.")
        return
    text = clean_command(text) or text
    process_command(text)


def deliver_reply(user_text, reply):
    conversation_history.append(make_content("model", reply))
    print(f"JARVIS: {reply}\n")
    threading.Thread(target=memory.save, args=(user_text, reply), daemon=True).start()
    speak(reply, already_printed=True)


def is_vision_request(text: str) -> bool:
    """True when the user is asking JARVIS to look at something via the camera."""
    t = text.lower().strip()
    phrases = (
        "what's this", "whats this", "what is this",
        "what am i holding", "what i'm holding", "what i am holding",
        "what am i doing", "what i'm doing", "what i am doing",
        "what am i up to", "describe what i'm doing", "describe what i am doing",
        "look at this", "look at that", "look at what", "look at me",
        "what do you see", "what can you see",
        "what is this object", "what's this object", "whats this object",
        "identify this", "describe this", "describe what i'm holding",
        "what am i showing", "take a look", "watch me",
    )
    return any(p in t for p in phrases)


def handle_vision_detection(description):
    """Legacy callback for optional CameraWatcher — not used on startup."""
    global is_processing
    if is_processing or stop_listening:
        return
    is_processing = True
    try:
        print(f"[Vision] JARVIS sees: {description}")
        speak(description)
        threading.Thread(
            target=memory.save,
            args=("[camera detection]", description),
            daemon=True,
        ).start()
    finally:
        is_processing = False


def process_command(text):
    global stop_listening, conversation_history, memory

    print(f"You said: {text}")

    handled, local_result = handle_locally(text)
    if handled:
        if local_result == "__EXIT__":
            speak("Going offline. Goodbye.")
            stop_listening = True
            return
        print("[Local] handled without Gemini API")
        conversation_history.append(make_content("user", text))
        deliver_reply(text, local_result)
        return

    # Check for exit keywords
    if any(word in text.lower() for word in ("goodbye", "quit", "exit", "bye", "stop")):
        speak("Going offline. Goodbye.")
        stop_listening = True
        return

    # On-demand camera vision (wake-word command only — camera opens for this call alone)
    if VISION_ENABLED and is_vision_request(text):
        print("[Vision] On-demand capture (camera opens briefly, faces blurred before upload)")
        speak("One moment — looking now.")
        conversation_history.append(make_content("user", text))
        try:
            description = capture_and_describe_once(client)
            deliver_reply(text, description)
        except Exception as e:
            print(f"Error in on-demand vision: {e}")
            speak("I couldn't use the camera just now. Please try again.")
        return

    # Check for intents that should be handled by the smart agent (weather, email, search)
    text_lower = text.lower()
    weather_keywords = ["weather", "forecast", "temperature", "rain", "sun", "cloud"]
    email_keywords = ["email", "send mail", "send email", "mail"]
    search_keywords = ["search", "look up", "google", "find", "search for"]

    is_weather = any(word in text_lower for word in weather_keywords)
    is_email = any(word in text_lower for word in email_keywords)
    is_search = any(word in text_lower for word in search_keywords)

    if is_weather or is_email or is_search:
        print("[SmartAgent] Detected intent for weather/email/search")
        try:
            reply = run_smart_agent(text, conversation_history)
            conversation_history.append(make_content("user", text))
            deliver_reply(text, reply)
        except Exception as e:
            print(f"Error in smart agent: {e}")
            speak(f"I encountered an error while processing your request: {str(e)}")
        return

    # If not handled by smart agent, fall back to the existing system-control path
    conversation_history.append(make_content("user", text))

    try:
        system_instruction = memory.build_system_instruction(
            text,
            conversation_history=conversation_history,
        ) + SYSTEM_CONTROL_PROMPT
        reply = generate_with_tools(client, conversation_history, system_instruction)
        deliver_reply(text, reply)
    except Exception as e:
        print(f"Error getting response: {e}")
        speak(friendly_error(e))

    time.sleep(0.3)


def listen_for_wake_word():
    """Listen continuously; on 'Jarvis' beep and record a dedicated command."""
    global is_processing, stop_listening

    print("👂 Listening for wake word 'Jarvis'...")
    audio_buffer = []
    silent_run = 0
    max_silent = int(WAKE_SILENCE_SECONDS / CHUNK_DURATION)
    min_samples = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while not stop_listening:
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk = chunk.flatten()

            if voice_detected(chunk):
                audio_buffer.extend(chunk)
                silent_run = 0
                continue

            if not audio_buffer:
                continue

            silent_run += 1
            if silent_run <= max_silent:
                continue

            if len(audio_buffer) < min_samples:
                audio_buffer = []
                silent_run = 0
                continue

            if is_processing or stop_listening:
                audio_buffer = []
                silent_run = 0
                continue

            transcript = transcribe(np.array(audio_buffer, dtype=np.float32))
            audio_buffer = []
            silent_run = 0

            if not extract_wake_word(transcript):
                continue

            print(f"🎯 Wake word detected! (heard: {transcript})")
            beep()
            _set_listening_state(True)
            is_processing = True
            try:
                inline = clean_command(transcript)
                if inline and len(inline.split()) >= 2:
                    _set_listening_state(False)
                    process_command(inline)
                else:
                    listen_for_command()
            finally:
                is_processing = False
                _set_listening_state(False)


def start_voice_interaction():
    global stop_listening, conversation_history, memory

    from security_checks import check_bitlocker_status
    check_bitlocker_status()

    widget_process = _launch_widget()
    _get_output_device()  # print chosen speaker once at startup

    calibrate_mic()
    print("Loading memory (SQLite + ChromaDB)...")
    memory = JarvisMemory()

    from system_control import set_gemini_client
    set_gemini_client(client)

    recent_chats = memory.get_recent()
    conversation_history = []
    for chat in recent_chats:
        conversation_history.append(make_content("user", chat["user"]))
        conversation_history.append(make_content("model", chat["jarvis"]))

    stop_listening = False
    listen_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    listen_thread.start()

    # Camera is on-demand only (capture_and_describe_once). No always-on watcher.
    if VISION_ENABLED:
        print("📷 Camera vision ready (on-demand — say 'Jarvis, what's this?')")
    else:
        print("📷 Camera vision disabled via .env")

    print("🗣️  Say 'Jarvis' then your command (pause briefly, then speak)")
    print("💡 Simple commands (open apps, volume) work even if API limit is hit")
    print("💡 Say 'goodbye' to stop")

    try:
        while not stop_listening:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        stop_listening = True
        listen_thread.join(timeout=1.0)
        _set_speaking_state(False)
        _set_listening_state(False)
        if widget_process is not None:
            try:
                widget_process.terminate()
            except Exception:
                pass


def _launch_widget():
    """Launch the floating visualizer as a detached process. Never blocks/crashes startup."""
    widget_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_widget.py")
    if not os.path.exists(widget_path):
        return None

    _set_speaking_state(False)
    _set_listening_state(False)

    # Prefer pythonw.exe (no console window); fall back to current interpreter
    py_dir = os.path.dirname(sys.executable)
    pythonw = os.path.join(py_dir, "pythonw.exe")
    interpreter = pythonw if os.path.exists(pythonw) else sys.executable

    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            [interpreter, widget_path],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("🌀 JARVIS visualizer widget launched")
        return proc
    except Exception as e:
        print(f"(Visualizer widget not started: {e})")
        return None


if __name__ == "__main__":
    start_voice_interaction()
