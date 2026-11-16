"""Look and feel: colours and fonts (`Theme`) applied through one Qt stylesheet.

To change the look, define another `Theme` and pass it to `apply_theme()` in app.py.
Widgets are styled by object name (`QLabel#title`, `QPushButton#primary`, ...), set in pages.py.
"""

from dataclasses import asdict, dataclass
from string import Template

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    background: str
    text: str
    subtle_text: str
    accent: str  # main buttons and the score
    accent_hover: str
    accent_pressed: str
    on_accent: str  # text on accent-coloured buttons
    warning: str  # "no face detected" messages


LIGHT = Theme(
    background="#faf6f0",
    text="#2e2a25",
    subtle_text="#857c70",
    accent="#f26b4f",
    accent_hover="#f5826a",
    accent_pressed="#d9583e",
    on_accent="#ffffff",
    warning="#c2410c",
)

FONT_FAMILIES = ["Noto Sans CJK JP", "Noto Sans JP"]

_STYLESHEET = Template("""
QMainWindow { background-color: $background; }
QWidget { color: $text; font-size: 18pt; }

QLabel#title { font-size: 32pt; font-weight: 700; }
QLabel#instruction { font-size: 26pt; font-weight: 700; }
QLabel#hint { color: $subtle_text; }
QLabel#caption { font-size: 16pt; font-weight: 700; color: $subtle_text; }
QLabel#status { font-weight: 700; color: $warning; }
QLabel#score { color: $accent; }
QLabel#message { font-size: 28pt; font-weight: 700; }
QLabel#closing { font-size: 34pt; font-weight: 700; }

QPushButton#primary {
    background-color: $accent;
    color: $on_accent;
    border: none;
    border-radius: 36px;
    min-height: 72px;
    min-width: 280px;
    padding: 0 48px;
    font-size: 24pt;
    font-weight: 700;
}
QPushButton#primary:hover { background-color: $accent_hover; }
QPushButton#primary:pressed { background-color: $accent_pressed; }
""")


def apply_theme(app: QApplication, theme: Theme = LIGHT) -> None:
    app.setStyle("Fusion")  # same base look on KDE and Windows
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    app.setFont(font)
    app.setStyleSheet(_STYLESHEET.substitute(asdict(theme)))
