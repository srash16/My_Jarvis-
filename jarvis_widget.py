"""
JARVIS floating visualizer widget (separate Qt process).

A frameless, always-on-top, semi-transparent glowing orb whose rings pulse
outward/inward. It reacts to JARVIS state via small local files:
  - speaking  -> strong, fast pulsing (lots of movement)
  - idle      -> gentle, slow pulsing (small movement)
  - listening -> grows in size (wake word heard, capturing your command)

Run standalone to check size/opacity/animation before wiring into the app:
    D:\\Srash-jarvis\\venv\\Scripts\\python.exe jarvis_widget.py

Missing state files are treated as idle (no crash).
"""

import json
import math
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMovie, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QMenu, QWidget

# ---- Tunables ----
CM_SIZE = 4.5              # base orb diameter in centimeters
WIDGET_OPACITY = 0.90      # 0..1 (higher = less transparent)
FALLBACK_PX_PER_CM = 37.8  # 96 DPI ≈ 37.8 px/cm

LISTEN_SCALE = 1.32        # how much bigger the orb gets while listening
MAX_SCALE = 1.55           # window is sized for this (largest possible orb)

IDLE_AMP = 0.030           # gentle breathing when idle
SPEAK_AMP = 0.135          # strong ring push when speaking
IDLE_SPEED = 1.4           # rad/s (slow)  -> ~4.5s period
SPEAK_SPEED = 5.2          # rad/s (fast)  -> ~1.2s period

FPS = 33
STATE_POLL_MS = 100
POSITION_SAVE_DEBOUNCE_MS = 500

BASE_DIR = Path(__file__).resolve().parent
PNG_PATH = BASE_DIR / "assets" / "jarvis_orb.png"
GIF_PATH = BASE_DIR / "assets" / "jarvis_orb.gif"
SPEAKING_PATH = BASE_DIR / "jarvis_speaking_state.json"
LISTENING_PATH = BASE_DIR / "jarvis_listening_state.json"
POSITION_PATH = BASE_DIR / "widget_position.json"


class JarvisWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # keep off the taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowOpacity(WIDGET_OPACITY)

        self.base_diam = self._compute_base_diameter()
        self.win_size = round(self.base_diam * MAX_SCALE)
        self.resize(self.win_size, self.win_size)

        self.base_pixmap = self._load_pixmap()

        # animation state
        self._phase = 0.0
        self._base_scale = 1.0       # smoothed (grows toward LISTEN_SCALE)
        self._last_t = time.monotonic()
        self._speaking = False
        self._listening = False
        self._state_accum = 0.0

        self._drag_offset = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_position)

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start(int(1000 / FPS))

        self._restore_position()

    # ---- sizing ----
    def _compute_base_diameter(self) -> int:
        try:
            screen = QApplication.primaryScreen()
            dpi = screen.logicalDotsPerInch() if screen else 0
            px_per_cm = (dpi / 2.54) if dpi and dpi > 0 else FALLBACK_PX_PER_CM
        except Exception:
            px_per_cm = FALLBACK_PX_PER_CM
        return max(48, round(CM_SIZE * px_per_cm))

    def _load_pixmap(self) -> QPixmap:
        if PNG_PATH.exists():
            pm = QPixmap(str(PNG_PATH))
            if not pm.isNull():
                return pm
        # Fallback: first frame of the old GIF, if present
        if GIF_PATH.exists():
            movie = QMovie(str(GIF_PATH))
            movie.jumpToFrame(0)
            pm = movie.currentPixmap()
            if not pm.isNull():
                return pm
        return QPixmap()  # null -> placeholder text painted instead

    # ---- state reading ----
    def _read_flag(self, path: Path, key: str) -> bool:
        try:
            if not path.exists():
                return False
            with open(path, "r", encoding="utf-8") as f:
                return bool(json.load(f).get(key, False))
        except Exception:
            return False

    def _poll_state(self):
        self._speaking = self._read_flag(SPEAKING_PATH, "speaking")
        self._listening = self._read_flag(LISTENING_PATH, "listening")

    # ---- animation ----
    def _tick(self):
        now = time.monotonic()
        dt = min(0.1, now - self._last_t)
        self._last_t = now

        self._state_accum += dt
        if self._state_accum >= STATE_POLL_MS / 1000.0:
            self._state_accum = 0.0
            self._poll_state()

        speed = SPEAK_SPEED if self._speaking else IDLE_SPEED
        amp = SPEAK_AMP if self._speaking else IDLE_AMP
        self._phase += speed * dt

        target_base = LISTEN_SCALE if self._listening else 1.0
        # smooth grow/shrink (~0.15s)
        self._base_scale += (target_base - self._base_scale) * min(1.0, dt * 8.0)

        self._current_scale = self._base_scale * (1.0 + amp * math.sin(self._phase))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self.base_pixmap.isNull():
            painter.setPen(Qt.cyan)
            painter.drawText(self.rect(), Qt.AlignCenter, "Missing:\njarvis_orb.png")
            return

        diam = max(1, int(self.base_diam * getattr(self, "_current_scale", 1.0)))
        scaled = self.base_pixmap.scaled(
            diam, diam, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

    # ---- position persistence ----
    def _restore_position(self):
        pos = None
        try:
            if POSITION_PATH.exists():
                with open(POSITION_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    pos = (int(data["x"]), int(data["y"]))
        except Exception:
            pos = None

        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen else None

        if pos is None and geo is not None:
            margin = 24
            pos = (geo.right() - self.win_size - margin, geo.bottom() - self.win_size - margin)

        if pos is not None:
            x, y = pos
            if geo is not None:
                x = min(max(x, geo.left()), geo.right() - self.win_size)
                y = min(max(y, geo.top()), geo.bottom() - self.win_size)
            self.move(x, y)

    def _save_position(self):
        try:
            with open(POSITION_PATH, "w", encoding="utf-8") as f:
                json.dump({"x": self.x(), "y": self.y()}, f)
        except Exception:
            pass

    # ---- dragging ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_offset)
            self._save_timer.start(POSITION_SAVE_DEBOUNCE_MS)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = None
            self._save_timer.start(POSITION_SAVE_DEBOUNCE_MS)
            event.accept()

    # ---- right-click menu ----
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        quit_action = menu.addAction("Quit")
        chosen = menu.exec_(event.globalPos())
        if chosen == quit_action:
            self._save_position()
            QApplication.quit()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    widget = JarvisWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
