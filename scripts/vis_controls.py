from typing import Any, Callable

from rich import print as rich_print

# Keyboard controls (press-based only)
KEYCODE_RIGHT = 262  # GLFW
KEYCODE_LEFT = 263
KEYCODE_PAGE_UP = 266
KEYCODE_PAGE_DOWN = 267
KEYCODE_SPACE = ord(" ")
KEYCODE_COMMA = ord(",")
KEYCODE_PERIOD = ord(".")
KEYCODE_MINUS = ord("-")
KEYCODE_EQUAL = ord("=")
KEYCODE_Z = ord("z")
KEYCODE_X = ord("x")
KEYCODE_Z_UPPER = ord("Z")
KEYCODE_X_UPPER = ord("X")

CLIP_STEP = 1
MULTIPLIER_MIN = 1
MULTIPLIER_MAX = 1000
MULTIPLIER_SCALE = 10
SPEED_MIN = 0.125
SPEED_MAX = 8.0

_PRINTED_HELP_VARIANTS: set[bool] = set()


def viewer_alive(viewer: Any) -> bool:
    if hasattr(viewer.viewer, "is_running"):
        return viewer.viewer.is_running()
    return True


def create_control_state(*, enable_curation: bool = False) -> dict[str, Any]:
    state: dict[str, Any] = {
        "clip_delta": 0,
        "paused": False,
        "frame_step": 0,
        "multiplier": MULTIPLIER_MIN,
        "speed": 1.0,
        "speed_dirty": False,
    }
    if enable_curation:
        state["curation_action"] = None
    return state


def controls_help_text(*, include_curation: bool = False) -> str:
    lines = [
        "[bold cyan]Keyboard Controls[/bold cyan]",
        f"[white]* Left/Right[/white] : change clip by current multiplier",
        f"[white]* PageUp/PgDn[/white] : multiplier x{MULTIPLIER_SCALE}/div{MULTIPLIER_SCALE} "
        f"(min {MULTIPLIER_MIN}, max {MULTIPLIER_MAX})",
        "[white]* - / =[/white] : playback speed 1/2x, 2x",
        "[white]* v[/white] : toggle CoM projection",
        "[white]* Space[/white] : pause/resume",
        "[white]* < / >[/white] : prev/next frame (paused only)",
    ]
    if include_curation:
        lines.append("[white]z / x[/white] : add/remove current clip in curation")
    return "\n".join(lines)


def log_controls_once(
    *, include_curation: bool = False, log_fn: Callable[[str], None] = rich_print
) -> None:
    if include_curation in _PRINTED_HELP_VARIANTS:
        return
    log_fn(controls_help_text(include_curation=include_curation))
    _PRINTED_HELP_VARIANTS.add(include_curation)


def make_keyboard_callback(
    state: dict[str, Any],
    *,
    enable_curation: bool = False,
    log_fn: Callable[[str], None] = rich_print,
):
    def callback(keycode: int, *args, **kwargs) -> None:
        if keycode == KEYCODE_RIGHT:
            state["clip_delta"] += CLIP_STEP * state["multiplier"]
        elif keycode == KEYCODE_LEFT:
            state["clip_delta"] -= CLIP_STEP * state["multiplier"]
        elif keycode == KEYCODE_PAGE_UP:
            state["multiplier"] = min(
                MULTIPLIER_MAX, state["multiplier"] * MULTIPLIER_SCALE
            )
            log_fn(f"[cyan]Multiplier: {state['multiplier']}[/cyan]")
        elif keycode == KEYCODE_PAGE_DOWN:
            state["multiplier"] = max(
                MULTIPLIER_MIN, state["multiplier"] // MULTIPLIER_SCALE
            )
            log_fn(f"[cyan]Multiplier: {state['multiplier']}[/cyan]")
        elif keycode == KEYCODE_SPACE:
            state["paused"] = not state["paused"]
        elif keycode == KEYCODE_COMMA and state["paused"]:
            state["frame_step"] = -1
        elif keycode == KEYCODE_PERIOD and state["paused"]:
            state["frame_step"] = 1
        elif keycode == KEYCODE_MINUS:
            state["speed"] = max(SPEED_MIN, state["speed"] * 0.5)
            state["speed_dirty"] = True
            log_fn(f"[cyan]Playback speed: {state['speed']:.3g}x[/cyan]")
        elif keycode == KEYCODE_EQUAL:
            state["speed"] = min(SPEED_MAX, state["speed"] * 2.0)
            state["speed_dirty"] = True
            log_fn(f"[cyan]Playback speed: {state['speed']:.3g}x[/cyan]")
        elif enable_curation and keycode in (KEYCODE_Z, KEYCODE_Z_UPPER):
            state["curation_action"] = "add"
        elif enable_curation and keycode in (KEYCODE_X, KEYCODE_X_UPPER):
            state["curation_action"] = "remove"

    return callback
