"""
vision.py - On-demand camera capture for JARVIS (privacy-first)

Default path: capture_and_describe_once() — opens the webcam only long enough
to grab a frame, blurs any detected faces locally, sends that frame to Gemini,
then releases the camera.

CameraWatcher (always-on motion loop) is kept for optional/manual use but is
NOT started by Jarvisgen on launch.
"""

import threading
import time

import cv2
import numpy as np
from google.genai import types

# ---- Tunables ----
MOTION_THRESHOLD = 25
MOTION_AREA_PERCENT = 2.0
STABLE_FRAMES_NEEDED = 8
COOLDOWN_SECONDS = 6
CAMERA_INDEX = 0
PREVIEW_WINDOW = "JARVIS Vision - Camera"
FACE_PAD = 0.30          # expand face box by 30% before blur
FACE_BLUR_SIGMA = 25     # strong Gaussian blur (privacy)

VISION_PROMPT = (
    "You are JARVIS's vision system. Look at this webcam frame and briefly "
    "describe, in one or two sentences, what the person is holding up or "
    "doing. If it's an object, name it and mention anything notable. "
    "If it's an action, describe the action. Keep it short and natural, "
    "like a witty AI assistant reacting in real time. "
    "Any faces in the image are intentionally blurred for privacy — "
    "ignore blurred regions and focus on objects or actions."
)

_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError(f"Could not load face cascade: {cascade_path}")
    return _face_cascade


def blur_faces(frame):
    """
    Return a COPY of frame with detected faces heavily Gaussian-blurred.
    Does not modify the original frame (safe for any local preview).
    """
    out = frame.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    h, w = out.shape[:2]
    for (x, y, fw, fh) in faces:
        pad_x = int(fw * FACE_PAD)
        pad_y = int(fh * FACE_PAD)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        k = max(31, int(min(roi.shape[0], roi.shape[1]) * 0.6) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), FACE_BLUR_SIGMA)
    return out


def capture_and_describe_once(client, prompt: str | None = None, camera_index: int = CAMERA_INDEX) -> str:
    """
    On-demand capture: open camera → grab frame → release camera →
    blur faces on an upload copy → ask Gemini → return description.

    Camera stays open only for this call (a few seconds).
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return "I couldn't open the camera. Check that it's connected and not in use."

    frame = None
    try:
        # Warm up auto-exposure with a couple of discarded reads
        for _ in range(3):
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
        if frame is None:
            ok, frame = cap.read()
        if not ok or frame is None:
            return "I couldn't capture a frame from the camera."
    finally:
        cap.release()

    print("[Vision] Camera closed — blurring faces before upload...")
    upload_frame = blur_faces(frame)

    ok, buf = cv2.imencode(".jpg", upload_frame)
    if not ok:
        return "I captured a frame but failed to encode it."

    image_bytes = buf.tobytes()
    use_prompt = (prompt or "").strip() or VISION_PROMPT

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                use_prompt,
            ],
        )
        description = (response.text or "").strip()
        return description or "I looked, but couldn't describe what I saw."
    except Exception as e:
        print(f"Vision analysis error: {e}")
        return f"Vision analysis failed: {e}"


class CameraWatcher:
    """
    Optional always-on motion watcher (NOT started by Jarvisgen by default).
    Faces are blurred on the upload copy only; local preview stays unblurred.
    """

    def __init__(self, client, on_detection, camera_index=CAMERA_INDEX, show_preview=True):
        self.client = client
        self.on_detection = on_detection
        self.camera_index = camera_index
        self.show_preview = show_preview
        self.stop_flag = False
        self.thread = None
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
        """Local preview only — faces NOT blurred (never leaves the machine)."""
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

        cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            preview, status, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "Press Q to close preview (local only — faces not blurred here)",
            (12, preview.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
        return preview

    def _watch_loop(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Could not open camera. Vision watcher disabled.")
            return

        print("Vision watcher active (optional always-on mode)")
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
                    if key in (ord("q"), ord("Q")):
                        preview_open = False
                        cv2.destroyWindow(PREVIEW_WINDOW)
                    try:
                        if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            preview_open = False
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
        print("Something held steady - analyzing (faces blurred for upload)...")
        self._analyzing = True
        try:
            upload_frame = blur_faces(frame)
            ok, buf = cv2.imencode(".jpg", upload_frame)
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
