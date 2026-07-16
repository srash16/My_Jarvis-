# JARVIS — Voice-Controlled AI Desktop Assistant

A hands-free AI assistant for Windows, inspired by Tony Stark’s JARVIS. You say **"Jarvis"**, speak a command, and the system listens, understands, remembers, answers, and can control the PC.

The project covers a full voice pipeline, an LLM brain with tool use, dual-layer memory (SQLite + ChromaDB RAG), desktop automation, and polish for reliability and lower-latency wake detection.

---

## What the Project Covers

### 1. Voice input & wake activation
- Continuous microphone streaming (16 kHz mono, 100 ms chunks via `sounddevice`)
- **Voice activity detection (VAD)** using RMS energy, with ambient-noise calibration at startup
- Silence-based endpointing (wake utterance and command recording end when you stop talking)
- Local **OpenAI Whisper** (`base` model) for speech-to-text on CPU
- Audio peak normalization and Whisper `initial_prompt` bias toward command vocabulary
- Wake phrase **"Jarvis"** detected via word-boundary match on the transcript
- Optional dedicated wake-word engine (e.g. Porcupine / openWakeWord) for faster, lighter always-on detection without full Whisper on every utterance
- Beep feedback when activated; record-until-silence for the follow-up command (up to ~12 s)

### 2. Speech output
- Primary TTS: **Microsoft Edge TTS** (`en-GB-RyanNeural`) — free neural voice
- Fallback TTS: **ElevenLabs** (`eleven_flash_v2_5`, free-tier premade voices such as Adam / Bella)
- Markdown stripping so spoken replies sound natural
- Playback through `soundfile` + `sounddevice` on the default output device
- Interruptible / barge-in style listening (stop speaking when the user talks again) as part of the completed voice UX

### 3. AI brain
- **Google Gemini 2.5 Flash** via the `google-genai` Python SDK
- Conversational persona (concise, slightly witty assistant style)
- Multi-turn chat history sent as Gemini `contents` (role + `parts` format)
- **Function calling / tool use**: model can request PC actions; the app executes tools and returns results in a multi-round loop
- Automatic retry and clear handling when the Gemini free-tier quota (429) is hit
- Local **regex intent router** for common commands (open apps, Google sites, volume, etc.) so those actions work without spending API quota

### 4. Memory & RAG
- **SQLite** (`jarvis_memory.db`) as the source of truth — every user/JARVIS exchange with timestamp
- **ChromaDB** persistent vector store (`chroma_data/`) for semantic search
- Local embeddings: **all-MiniLM-L6-v2** (384 dimensions, ONNX via Chroma’s default embedding function)
- **Turn-level chunking**: one chunk = one full Q&A pair (`User: …` / `JARVIS: …`)
- **Hybrid conversational RAG**:
  - Short-term: last N recent turns loaded into the model context
  - Long-term: top-K similar past exchanges retrieved by cosine similarity (HNSW index) and injected into the system instruction
- Startup sync: SQLite rows missing from Chroma are re-indexed automatically
- Async save after each reply so TTS is not blocked
- Stronger retrieval options in the full build: re-ranking, hybrid BM25 + dense search, and metadata filters (e.g. by date)

### 5. System control (desktop automation)
Twenty-plus tools exposed to Gemini (and many also reachable via the local router):

| Area | Capabilities |
|------|----------------|
| Apps | Open Chrome, Notepad, VS Code, Cursor, Spotify, Terminal, Settings, etc.; custom app paths from `.env` |
| Chrome | Open a specific **Google account profile** by email, partial match, or nickname; open URLs in that profile |
| Web | Open websites; Google services (Classroom, Drive, Gmail, Meet, YouTube, …) mapped to URLs (not fake Windows executables) |
| Files | List / open / create folders; delete & move with confirmation — restricted to the user home directory |
| Vision | Screenshot + Gemini multimodal analysis (“what’s on my screen?”) |
| Audio / display | Get/set/mute volume (pycaw); set brightness (WMI, laptops) |
| Power | Lock, sleep, shutdown/restart with verbal confirmation and cancel window |
| Input / windows | Type text, hotkeys, click coordinates; list and focus windows (PyAutoGUI / PyGetWindow) |

Safety: home-folder sandbox, two-step confirmation for destructive actions, PyAutoGUI failsafe (mouse to top-left corner aborts automation).

### 6. Reliability & product polish
- Mic calibration and clearer “didn’t catch that” recovery
- TTS error handling so a voice failure does not kill the listener thread
- Config via `.env` (API keys, Chrome nicknames, custom apps, RAG knobs, power delay)
- Optional simple UI or memory/log dashboard
- Web search / browsing tools for live information
- Packaging and day-to-day stability improvements

---

## Architecture

```
Microphone
    → VAD (RMS) + silence endpointing
    → Whisper STT (local)
    → Wake word check ("Jarvis")
    → Command text
         ├─ Local intent router (regex) ──→ System tools ──→ TTS
         └─ RAG (recent SQLite turns + Chroma top-K)
                → Gemini 2.5 Flash (+ function calling)
                → Tool results (if any)
                → Final reply → Edge TTS / ElevenLabs → Speakers
                → Save exchange → SQLite + ChromaDB (background)
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language / OS | Python 3.13 · Windows 10/11 |
| STT | openai-whisper (`base`), NumPy, sounddevice, soundfile |
| LLM | google-genai · Gemini 2.5 Flash (text + vision) |
| TTS | edge-tts · ElevenLabs |
| Memory | SQLite · ChromaDB · Sentence Transformers MiniLM-L6-v2 |
| Automation | PyAutoGUI · PyGetWindow · pycaw · comtypes · Pillow · subprocess / WMI |
| Config | python-dotenv |

---

## Algorithms & Techniques (detail)

### Voice activity detection
Each audio chunk’s energy is:

\[ E = \sqrt{\frac{1}{N}\sum_i x_i^2} \]

Compared to a threshold set from a short ambient recording: roughly `max(0.004, noise_rms × 2.5)`.

### Endpoint detection
Voiced chunks fill a buffer; a run of silence (about 0.8 s for wake, 1.2 s for commands) closes the utterance. Very short buffers are discarded as noise.

### Wake word
Transcribe the utterance, then match `\bjarvis\b`. The completed project also supports a dedicated wake-word model so the always-on path does not need full ASR on every sound.

### Local intent routing
Ordered regex rules for Google apps, Chrome-with-profile, known desktop apps, volume/brightness/power, etc. Prevents treating words like “google” as Windows executables.

### LLM tool loop
Gemini is called with tools declared from Python functions (type hints + docstrings). Automatic function calling is disabled; the app runs a controlled loop: detect `function_call` → execute → send `function_response` → repeat until a final text answer (capped rounds).

### RAG retrieval
Query is embedded with the same MiniLM model; Chroma returns the nearest conversation chunks by cosine distance. Results are formatted into the system prompt alongside the persona. Optional re-ranking and hybrid lexical+dense search improve precision on larger histories.

---

## Project Structure

```
Jarvisgen.py         # Main app: audio I/O, VAD, wake word, Whisper, TTS, loop
jarvis_brain.py      # Gemini client, tool-calling loop, retries / quota messages
memory.py            # SQLite + ChromaDB dual store and RAG retrieval
rag_config.py        # Embedding model, top-K, chunk template, persona (env-overridable)
system_control.py    # Desktop tools, Chrome profiles, safety guards
local_commands.py    # Zero-API command matching
system_config.py     # Chrome nicknames, custom apps, power delay from .env
micetest.py          # Microphone diagnostic helper
```

Runtime data: `jarvis_memory.db`, `chroma_data/`, optional `~/.jarvis_screenshots/`.

---

## Setup

```powershell
cd D:\Srash-jarvis
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install google-genai openai-whisper elevenlabs edge-tts chromadb `
  sounddevice soundfile numpy python-dotenv pyautogui pygetwindow pycaw comtypes pillow
```

`.env` example:

```env
GOOGLE_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
ELEVENLABS_VOICE_ID=pNInz6obpgDQGcFmaJgB
JARVIS_CHROME_NICKNAMES=work=mmcoe.edu.in;personal=you@gmail.com
JARVIS_CUSTOM_APPS=obs=C:\Path\To\obs64.exe
JARVIS_POWER_DELAY=30
RAG_SEMANTIC_TOP_K=5
RAG_RECENT_TURNS=5
```

```powershell
python Jarvisgen.py
```

First run downloads Whisper (~140 MB) and the MiniLM embedding model (~80 MB) once.

---

## Example Commands

- *"Jarvis, open notepad"*
- *"Jarvis, open Chrome with work account"*
- *"Jarvis, open Google Classroom on mmcoe profile"*
- *"Jarvis, set volume to 50"*
- *"Jarvis, what's on my screen?"*
- *"Jarvis, what did we talk about last time regarding memory?"*
- *"Jarvis, lock my computer"* / *"Jarvis, goodbye"*

---

## Safety

- File tools only under the user home directory  
- Delete / overwrite / shutdown require explicit verbal confirmation  
- Shutdown uses a delay window and can be cancelled  
- PyAutoGUI failsafe: move mouse to the top-left corner to abort automation  
- Secrets stay in `.env` (gitignored)

---

## Notes

Educational project integrating speech processing, RAG, LLM agents, and Windows automation. Gemini API usage is subject to provider rate limits; local routing keeps common PC commands usable when the API is throttled.
