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



print(os.getenv("GOOGLE_API_KEY"))

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
VOICE_ID=os.getenv("VOICEID_API_KEY")
print("Loading Whisper..... (first run downloads 140 MB, one time only )")
whisper_model = whisper.load_model("base")
print("JARVIS is online..... Type 'quit' to exit\n")

SAMPLE_RATE = 16000
DURATION=10 #SECONDS

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
    with tempfile.TemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(f.name,audio_array, SAMPLE_RATE)
    result=whisper_model.transcribe(temp_path)

    #return result["text"].strip()

def speak(text):
    print(f"JARVIS: {text}\n")
    audio = eleven.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_monolingual_v1"
    )
    play(audio)

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

#while True:
#   user_input = input("You: ").strip()
#
 #   if user_input.lower() == "quit":
  #      print("JARVIS: Okay, going offline. Goodbye.")
   #     break

    #if not user_input:
     #   continue

    #response = chat.send_message(user_input)
    #reply = response.text

    #print(f"\nJARVIS: {reply}\n")























    
##################################

#conversation_history = []
#print("JARVIS is online..... Type 'quit' to exit")

#while True:
 #   user_input = input("You: ").strip()

  #  if user_input.lower() == "quit":
   #     print("Okay going offline....")
   # if not user_input:
    #    continue

    #conversation_history.append(f"User: {user_input}")

    #prompt = f"""
#You are JARVIS, a helpful and intelligent AI assistant.
#Be concise, smart, and slightly witty.

#Conversation:
#{chr(10).join(conversation_history)}

#JARVIS:
#"""

 #   response = client.models.generate_content(
  #      model="gemini-2.5-flash",
   #     contents=prompt
    #)

    #reply = response.text

    #conversation_history.append(f"JARVIS: {reply}")

    #print(f"\nJARVIS: {reply}\n")