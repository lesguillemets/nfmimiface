"""Look and feel: colours and fonts (`Theme`) applied through one Qt stylesheet.

To add a look, define another `Theme` and list it in `THEMES` (chosen with `--theme`).
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
    surface: str  # image panels
    border: str  # hairlines around panels
    accent: str  # main buttons and the score
    accent_hover: str
    accent_pressed: str
    on_accent: str  # text on accent-coloured buttons
    warning: str  # "no face detected" messages
    button_radius: int  # px; half the button height (36) makes a pill
    image_radius: int  # px


LIGHT = Theme(
    background="#faf6f0",
    text="#2e2a25",
    subtle_text="#857c70",
    surface="#ffffff",
    border="#e6dfd4",
    accent="#f26b4f",
    accent_hover="#f5826a",
    accent_pressed="#d9583e",
    on_accent="#ffffff",
    warning="#c2410c",
    button_radius=36,
    image_radius=14,
)

DARK = Theme(  # navy #00205B with a gold accent
    background="#00205b",
    text="#f3f6fc",
    subtle_text="#9fb3da",
    surface="#0a2c6b",
    border="#1f4285",
    accent="#ffc53d",
    accent_hover="#ffd36e",
    accent_pressed="#e6a817",
    on_accent="#00205b",
    warning="#ff9a7a",
    button_radius=4,
    image_radius=4,
)

THEMES = {"light": LIGHT, "dark": DARK}

FONT_FAMILIES = ["Noto Sans CJK JP", "Noto Sans JP"]

_STYLESHEET = Template("""
QMainWindow, QDialog { background-color: $background; }
QWidget { color: $text; font-size: 18pt; }

ImageView {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: ${image_radius}px;
    qproperty-radius: $image_radius;
}

QLabel#title { font-size: 32pt; font-weight: 700; }
QLabel#instruction { font-size: 26pt; font-weight: 700; }
QLabel#hint { color: $subtle_text; }
QLabel#caption { font-size: 16pt; font-weight: 700; color: $subtle_text; }
QLabel#status { font-weight: 700; color: $warning; }
QLabel#score { color: $accent; }
QLabel#message { font-size: 28pt; font-weight: 700; }
QLabel#closing { font-size: 34pt; font-weight: 700; }

QPushButton {  /* plain buttons, e.g. OK in the error dialog */
    background-color: transparent;
    border: 2px solid $subtle_text;
    border-radius: 8px;
    padding: 6px 24px;
}
QPushButton#primary {
    background-color: $accent;
    color: $on_accent;
    border: none;
    border-radius: ${button_radius}px;
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
