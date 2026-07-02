import sounddevice as sd
import numpy as np
import soundfile as sf

SAMPLE_RATE=16000
DURATION = 6
print(sd.query_devices())
print("\nRecording for 6 seconds... SPEAK NOW")

audio = sd.rec(
 int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)
sd.wait()

volume = np.abs(audio).mean()
volume = np.abs(audio).mean()
print(f"\nAverage volume detected: {volume:.6f}")

if volume < 0.001:
    print("❌ Mic is picking up silence — wrong device or mic muted")
elif volume < 0.01:
    print("⚠️ Very quiet — mic works but speak louder")
else:
    print("✅ Mic is working fine")

sf.write("test_recording.wav", audio, SAMPLE_RATE)
print("Saved test_recording.wav — play it in File Explorer to hear what was captured")