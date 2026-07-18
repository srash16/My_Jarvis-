"""
vision.py - Camera-based object/action detection for JARVIS

Watches the webcam in a background thread. When something is held
steady in front of the camera (or a pause follows movement), it grabs
a frame, asks Gemini to describe what it sees, and hands the description
off to a callback (wired to speak() in Jarvisgen.py).

Shows a live preview window so you can see yourself and any object
in frame while JARVIS is watching.
"""

import threading
import time

import cv2
import numpy as np
from google.genai import types

# ---- Tunables ----
MOTION_THRESHOLD = 25        # pixel-diff sensitivity (lower = more sensitive)
MOTION_AREA_PERCENT = 2.0    # % of frame that must change to count as "motion"
STABLE_FRAMES_NEEDED = 8     # frames of stillness after motion = trigger
COOLDOWN_SECONDS = 6         # don't re-trigger immediately after describing
CAMERA_INDEX = 0
PREVIEW_WINDOW = "JARVIS Vision - Camera"

VISION_PROMPT = (
    "You are JARVIS's vision system. Look at this webcam frame and briefly "
    "describe, in one or two sentences, what the person is holding up or "
    "doing. If it's an object, name it and mention anything notable. "
    "If it's an action, describe the action. Keep it short and natural, "
    "like a witty AI assistant reacting in real time."
)


class CameraWatcher:
    def __init__(self, client, on_detection, camera_index=CAMERA_INDEX, show_preview=True):
        """
        client: your existing genai.Client instance
        on_detection: callback(description: str) - called when something is detected
        show_preview: if True, opens a live camera window so you can see yourself
        """
        self.client = client
        self.on_detection = on_detection
        self.camera_index = camera_index
        self.show_preview = show_preview
        self.stop_flag = False
        self.thread = None
        self._status = "Watching..."
        self._analyzing = False

    def start(self):
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=2.0)
        try:
            cv2.destroyWindow(PREVIEW_WINDOW)
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _draw_preview(self, frame, motion_active, in_cooldown, stable_count):
        """Mirror preview like a normal front camera, with status overlay."""
        preview = cv2.flip(frame, 1)

        if self._analyzing:
            status = "Analyzing..."
            color = (0, 165, 255)
        elif in_cooldown:
            status = "Cooldownoldown"
            color = (128, 128, 128)
        elif motion_active:
            status = f"Motion - hold steady ({stable_count}/{STABLE_FRAMES_NEEDED})"
            color = (0, 255, 255)
        else:
            status = "Watching - hold something up"
            color = (0, 255, 0)

        # Dark bar behind text for readability
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            preview,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "Press Q to close preview (vision keeps running)",
            (12, preview.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return preview

    def _watch_loop(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Could not open camera. Vision watcher disabled.")
            return

        print("Vision watcher active - live camera window opened")
        print("Hold something steady in front of the camera to trigger detection")

        if self.show_preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW, 960, 720)

        prev_gray = None
        motion_active = False
        stable_count = 0
        last_trigger_time = 0.0
        preview_open = self.show_preview

        try:
            while not self.stop_flag:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if prev_gray is None:
                    prev_gray = gray
                    if preview_open:
                        cv2.imshow(PREVIEW_WINDOW, self._draw_preview(frame, False, False, 0))
                        cv2.waitKey(1)
                    time.sleep(0.05)
                    continue

                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
                motion_percent = (np.count_nonzero(thresh) / thresh.size) * 100
                prev_gray = gray

                now = time.time()
                in_cooldown = (now - last_trigger_time) < COOLDOWN_SECONDS

                if motion_percent > MOTION_AREA_PERCENT:
                    motion_active = True
                    stable_count = 0
                elif motion_active and not in_cooldown:
                    stable_count += 1
                    if stable_count >= STABLE_FRAMES_NEEDED:
                        motion_active = False
                        stable_count = 0
                        last_trigger_time = now
                        # Analyze a copy so preview keeps updating
                        capture = frame.copy()
                        threading.Thread(
                            target=self._capture_and_describe,
                            args=(capture,),
                            daemon=True,
                        ).start()

                if preview_open:
                    display = self._draw_preview(frame, motion_active, in_cooldown, stable_count)
                    cv2.imshow(PREVIEW_WINDOW, display)
                    key = cv2.waitKey(1) & 0xFF
                    # Q closes preview only; watcher keeps running headless
                    if key in (ord("q"), ord("Q")):
                        preview_open = False
                        cv2.destroyWindow(PREVIEW_WINDOW)
                        print("Camera preview closed (vision still active in background)")
                    # Window X button
                    try:
                        if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            preview_open = False
                            print("Camera preview closed (vision still active in background)")
                    except Exception:
                        preview_open = False

                time.sleep(0.05)
        finally:
            cap.release()
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _capture_and_describe(self, frame):
        print("Something held steady - analyzing...")
        self._analyzing = True
        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                return
            image_bytes = buf.tobytes()

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    VISION_PROMPT,
                ],
            )
            description = (response.text or "").strip()
            if description:
                self.on_detection(description)
        except Exception as e:
            print(f"Vision analysis error: {e}")
        finally:
            self._analyzing = False
