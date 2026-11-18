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
    radius: int  # px, corners of buttons and image panels


LIGHT = Theme(  # cool paper white with the navy as accent
    background="#f3f4f6",
    text="#182133",
    subtle_text="#5b6577",
    surface="#ffffff",
    border="#d5dae1",
    accent="#00205b",
    accent_hover="#13377a",
    accent_pressed="#001538",
    on_accent="#ffffff",
    warning="#b42318",
    radius=3,
)

DARK = Theme(  # near-black with a muted gold accent
    background="#0b1019",
    text="#e9edf4",
    subtle_text="#8f9bb2",
    surface="#141b28",
    border="#253046",
    accent="#d4a437",
    accent_hover="#deb453",
    accent_pressed="#b98c27",
    on_accent="#0b1019",
    warning="#ff8f73",
    radius=3,
)

THEMES = {"light": LIGHT, "dark": DARK}

FONT_FAMILIES = ["Noto Sans CJK JP", "Noto Sans JP"]

_STYLESHEET = Template("""
QMainWindow, QDialog { background-color: $background; }
QWidget { color: $text; font-size: 16pt; }

ImageView {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: ${radius}px;
    qproperty-radius: $radius;
}

QLabel#title { font-size: 28pt; font-weight: 700; letter-spacing: 2px; }
QLabel#instruction { font-size: 22pt; font-weight: 500; }
QLabel#hint { color: $subtle_text; }
QLabel#caption { font-size: 13pt; font-weight: 700; letter-spacing: 4px; color: $subtle_text; }
QLabel#status { font-weight: 700; color: $warning; }
QLabel#score { color: $accent; }
QLabel#message { font-size: 22pt; font-weight: 500; }
QLabel#closing { font-size: 28pt; font-weight: 500; letter-spacing: 1px; }

QPushButton {  /* plain buttons, e.g. OK in the error dialog */
    background-color: transparent;
    border: 2px solid $subtle_text;
    border-radius: ${radius}px;
    padding: 6px 24px;
}
QPushButton#primary {
    background-color: $accent;
    color: $on_accent;
    border: none;
    border-radius: ${radius}px;
    min-height: 60px;
    min-width: 260px;
    padding: 0 48px;
    font-size: 18pt;
    font-weight: 700;
    letter-spacing: 4px;
}
QPushButton#primary:hover { background-color: $accent_hover; }
QPushButton#primary:pressed { background-color: $accent_pressed; }
QPushButton#primary:disabled { background-color: $border; color: $subtle_text; }
""")


def apply_theme(app: QApplication, theme: Theme = LIGHT) -> None:
    app.setStyle("Fusion")  # same base look on KDE and Windows
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    app.setFont(font)
    app.setStyleSheet(_STYLESHEET.substitute(asdict(theme)))
