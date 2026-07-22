"""
JARVIS floating HUD visualizer (separate Qt process).

Primary renderer: QWebEngineView + assets/jarvis_orb.html (SVG/CSS rings).
Fallback: QLabel + QMovie (jarvis_orb.gif) if QtWebEngine is unavailable.

Window: frameless, always-on-top, translucent, draggable, position saved.
State: polls jarvis_speaking_state.json / jarvis_listening_state.json.

Run standalone:
    D:\\Srash-jarvis\\venv\\Scripts\\python.exe jarvis_widget.py
"""

import json
import math
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QMovie, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMenu, QVBoxLayout, QWidget

# QtWebEngine is optional — many PyQt5 installs need: pip install PyQtWebEngine
WEBENGINE_AVAILABLE = False
QWebEngineView = None
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except Exception:
    pass

# ---- Tunables ----
CM_SIZE = 4.5
WIDGET_OPACITY = 0.98          # nearly opaque — rings stay bright
FALLBACK_PX_PER_CM = 37.8

LISTEN_SCALE = 1.28
MAX_SCALE = 1.45

# Legacy fallback pulse (GIF/PNG path only)
IDLE_AMP = 0.030
SPEAK_AMP = 0.135
IDLE_SPEED = 1.4
SPEAK_SPEED = 5.2

FPS = 33
STATE_POLL_MS = 100
POSITION_SAVE_DEBOUNCE_MS = 500

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "assets" / "jarvis_orb.html"
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
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowOpacity(WIDGET_OPACITY)

        self.base_diam = self._compute_base_diameter()
        self.win_size = round(self.base_diam * MAX_SCALE)
        self.resize(self.win_size, self.win_size)

        self._speaking = False
        self._listening = False
        self._last_speaking = None
        self._last_listening = None
        self._page_ready = False
        self.use_webengine = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if WEBENGINE_AVAILABLE and HTML_PATH.exists():
            try:
                self._setup_webengine(layout)
                self.use_webengine = True
                print("[Widget] Using QtWebEngine HUD (jarvis_orb.html)")
            except Exception as e:
                print(f"[Widget] QtWebEngine init failed ({e}); using GIF fallback")
        else:
            if not WEBENGINE_AVAILABLE:
                print(
                    "[Widget] PyQtWebEngine not installed — using GIF fallback. "
                    "Install with: pip install PyQtWebEngine"
                )
            elif not HTML_PATH.exists():
                print(f"[Widget] Missing {HTML_PATH.name} — using GIF fallback")

        if not self.use_webengine:
            self._setup_legacy(layout)

        self._drag_offset = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_position)

        self._state_timer = QTimer(self)
        self._state_timer.timeout.connect(self._poll_and_apply_state)
        self._state_timer.start(STATE_POLL_MS)

        if not self.use_webengine:
            self._phase = 0.0
            self._base_scale = 1.0
            self._current_scale = 1.0
            self._last_t = time.monotonic()
            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._tick_legacy)
            self._anim_timer.start(int(1000 / FPS))

        self._restore_position()
        self._poll_and_apply_state()

    # ---- sizing ----
    def _compute_base_diameter(self) -> int:
        try:
            screen = QApplication.primaryScreen()
            dpi = screen.logicalDotsPerInch() if screen else 0
            px_per_cm = (dpi / 2.54) if dpi and dpi > 0 else FALLBACK_PX_PER_CM
        except Exception:
            px_per_cm = FALLBACK_PX_PER_CM
        return max(48, round(CM_SIZE * px_per_cm))

    # ---- WebEngine HUD ----
    def _setup_webengine(self, layout):
        self.webview = QWebEngineView(self)
        self.webview.setAttribute(Qt.WA_TranslucentBackground, True)
        self.webview.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.webview.page().setBackgroundColor(Qt.transparent)
        self.webview.loadFinished.connect(self._on_page_ready)
        self.webview.load(QUrl.fromLocalFile(str(HTML_PATH.resolve())))
        layout.addWidget(self.webview)

    def _on_page_ready(self, ok: bool):
        self._page_ready = bool(ok)
        if ok:
            self._apply_web_state(force=True)

    def _apply_web_state(self, force: bool = False):
        if not self.use_webengine or not self._page_ready:
            return
        if not force and self._speaking == self._last_speaking and self._listening == self._last_listening:
            return
        self._last_speaking = self._speaking
        self._last_listening = self._listening

        state = "speaking" if self._speaking else "idle"
        scale = LISTEN_SCALE if self._listening else 1.0
        page = self.webview.page()
        page.runJavaScript(f"setState('{state}');")
        page.runJavaScript(f"setScale({scale:.4f});")

    # ---- Legacy GIF / PNG fallback ----
    def _setup_legacy(self, layout):
        self.legacy_label = QLabel(self)
        self.legacy_label.setAlignment(Qt.AlignCenter)
        self.legacy_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.legacy_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.legacy_label)

        self._movie = None
        self._fallback_pixmap = QPixmap()

        if GIF_PATH.exists():
            self._movie = QMovie(str(GIF_PATH))
            self._movie.setScaledSize(self.legacy_label.size())
            self.legacy_label.setMovie(self._movie)
            self._movie.start()
            self._movie.setPaused(True)
            print(f"[Widget] Legacy renderer: GIF ({GIF_PATH.name})")
        else:
            self._fallback_pixmap = self._load_pixmap()
            if self._fallback_pixmap.isNull():
                self.legacy_label.setText("Missing jarvis_orb.gif")
            print("[Widget] Legacy renderer: static PNG")

    def _load_pixmap(self) -> QPixmap:
        if PNG_PATH.exists():
            pm = QPixmap(str(PNG_PATH))
            if not pm.isNull():
                return pm
        return QPixmap()

    def _tick_legacy(self):
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - getattr(self, "_last_t", now)))
        self._last_t = now

        speed = SPEAK_SPEED if self._speaking else IDLE_SPEED
        amp = SPEAK_AMP if self._speaking else IDLE_AMP
        self._phase += speed * dt

        target_base = LISTEN_SCALE if self._listening else 1.0
        self._base_scale += (target_base - self._base_scale) * min(1.0, dt * 8.0)
        self._current_scale = self._base_scale * (1.0 + amp * math.sin(self._phase))

        if self._movie is not None:
            if self._speaking:
                self._movie.setPaused(False)
                if self._movie.state() != QMovie.Running:
                    self._movie.start()
            else:
                self._movie.setPaused(True)
        else:
            self.update()

    def paintEvent(self, event):
        if self.use_webengine:
            return
        if getattr(self, "_movie", None) is not None:
            return
        if getattr(self, "_fallback_pixmap", QPixmap()).isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        diam = min(
            int(self.base_diam * getattr(self, "_current_scale", 1.0)),
            self.width(),
            self.height(),
        )
        scaled = self._fallback_pixmap.scaled(
            max(1, diam), max(1, diam), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

    # ---- shared state ----
    def _read_flag(self, path: Path, key: str) -> bool:
        try:
            if not path.exists():
                return False
            with open(path, "r", encoding="utf-8") as f:
                return bool(json.load(f).get(key, False))
        except Exception:
            return False

    def _poll_and_apply_state(self):
        self._speaking = self._read_flag(SPEAKING_PATH, "speaking")
        self._listening = self._read_flag(LISTENING_PATH, "listening")
        if self.use_webengine:
            self._apply_web_state()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.use_webengine and self._movie is not None:
            self._movie.setScaledSize(self.legacy_label.size())

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
            pos = (
                geo.right() - self.win_size - margin,
                geo.bottom() - self.win_size - margin,
            )

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

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        quit_action = menu.addAction("Quit")
        chosen = menu.exec_(event.globalPos())
        if chosen == quit_action:
            self._save_position()
            QApplication.quit()


def main():
    # Required by QtWebEngine on many Windows setups (must be before QApplication)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    widget = JarvisWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
