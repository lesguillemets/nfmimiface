"""Radar charts of a session's results: the participant's blendshapes vs the stimulus's.

    uv run nfmimiface-radar SESSION_JSON [--include NAME ...] [--show]

Saves one PNG per trial next to the JSON (SESSION_ID_t1_radar.png, ...), or shows them
with --show. The axes are the blendshapes the session's scorer used, as far as its
params tell (`include`, else all but `exclude`); scorers without these params, such as
regions and emotions, get all but NON_EXPRESSIVE. --include overrides this.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.projections.polar import PolarAxes

from nfmimiface.face import BLENDSHAPE_NAMES
from nfmimiface.scoring import NON_EXPRESSIVE

TARGET_COLOR = "#2a78d6"  # blue
PARTICIPANT_COLOR = "#eb6834"  # orange


def axis_names(params: Mapping[str, Any]) -> list[str]:
    """Blendshapes the scorer with `params` uses, in BLENDSHAPE_NAMES order."""
    if (include := params.get("include")) is not None:
        return [n for n in BLENDSHAPE_NAMES if n in include]
    exclude = params.get("exclude", NON_EXPRESSIVE)
    return [n for n in BLENDSHAPE_NAMES if n not in exclude]


class Radar:
    """Radar chart of a stimulus and a participant on the axes `names` (0..1), drawn on
    `fig`. Each set_* call replaces one polygon (None hides it); the caller redraws."""

    def __init__(self, fig: Figure, names: Sequence[str]) -> None:
        self.names = list(names)
        angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
        self._closed = np.append(angles, angles[0])  # repeat the first point to close it
        fig.subplots_adjust(left=0.15, right=0.85, top=0.78, bottom=0.12)  # room for labels
        ax = cast(PolarAxes, fig.add_subplot(projection="polar"))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise
        ax.set_xticks(angles, [])
        for theta, name in zip(angles, names, strict=True):
            # Radial labels just outside the circle, kept upright on the left half.
            degrees = 90 - np.degrees(theta)  # theta runs clockwise from the top
            left = np.cos(np.radians(degrees)) < -1e-9
            ax.text(
                theta, 1.05, name, fontsize=8, va="center", ha="right" if left else "left",
                rotation=degrees + 180 if left else degrees, rotation_mode="anchor",
            )  # fmt: skip
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "0.5", "", "1"], fontsize=8, color="gray")
        ax.grid(color="lightgray", linewidth=0.5)
        ax.spines["polar"].set_color("lightgray")
        self._target = self._series(ax, "stimulus", TARGET_COLOR)
        self._participant = self._series(ax, "participant", PARTICIPANT_COLOR)
        fig.legend(loc="upper right")

    def _series(self, ax: PolarAxes, label: str, color: str) -> tuple[Line2D, Polygon]:
        zeros = np.zeros_like(self._closed)
        (line,) = ax.plot(self._closed, zeros, color=color, linewidth=2, label=label)
        (fill,) = ax.fill(self._closed, zeros, color=color, alpha=0.15)
        return line, fill

    def set_target(self, values: Mapping[str, float] | None) -> None:
        self._set(self._target, values)

    def set_participant(self, values: Mapping[str, float] | None) -> None:
        self._set(self._participant, values)

    def _set(self, series: tuple[Line2D, Polygon], values: Mapping[str, float] | None) -> None:
        line, fill = series
        line.set_visible(values is not None)
        fill.set_visible(values is not None)
        if values is not None:
            r = [values[n] for n in self.names]
            r.append(r[0])
            line.set_data(self._closed, r)
            fill.set_xy(np.column_stack([self._closed, r]))


def radar(
    names: Sequence[str], target: Mapping[str, float], participant: Mapping[str, float]
) -> Figure:
    """Radar chart of `target` and `participant` on the axes `names` (0..1)."""
    fig = plt.figure(figsize=(10, 10))
    chart = Radar(fig, names)
    chart.set_target(target)
    chart.set_participant(participant)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Radar charts: participant vs stimulus.")
    parser.add_argument("session", type=Path, help="session JSON")
    parser.add_argument("--include", nargs="+", metavar="NAME", help="blendshapes as axes")
    parser.add_argument("--show", action="store_true", help="show instead of saving")
    args = parser.parse_args()

    record = json.loads(args.session.read_text(encoding="utf-8"))
    scorer = record["scorer"]
    names = axis_names({"include": args.include} if args.include else scorer["params"])
    for i, trial in enumerate(record["trials"], start=1):
        fig = radar(names, trial["target_blendshapes"], trial["blendshapes"])
        fig.suptitle(
            f"{record['session_id']} trial {i}: {trial['stimulus']}\n"
            f"score {trial['score']:.1f} ({scorer['name']})"
        )
        if not args.show:
            out = args.session.with_name(f"{record['session_id']}_t{i}_radar.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
