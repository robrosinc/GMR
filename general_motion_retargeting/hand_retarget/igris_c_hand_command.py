"""Normalized IGRIS-C hand command: 12 motor targets in [0, 1], shared by sim and robot.

The real IGRIS-C hand (igris_c_hand) takes one normalized target per Dynamixel
motor (``rt/handcmd``: 0 = extension reference, 1 = flexion reference, motor
ticks interpolated linearly). This module defines that command as the common
finger representation for every consumer:

* ``MOTOR_IDS`` fixes the 12-slot order to the hardware motor IDs
  ``[11..16, 21..26]`` (right hand first, then left).
* ``hand_keypoints_to_normalized`` derives one hand's six values directly from
  the 26 OpenXR ``XR_EXT_hand_tracking`` keypoints carried by the VRS1 uplink
  (per-finger flexion angle sums and the thumb spread angle, normalized with
  open/closed calibration constants). Angles between bone vectors are invariant
  to the wrist frame, so no operator-to-robot rotation is needed.
* ``grip_to_normalized`` maps the controllers' analog grip onto the same slots.
* ``HandSourceArbiter`` picks hand tracking, controller grip, or hold per hand.
* ``normalized_to_joint_targets`` converts the 12 values to simulator joint
  targets (radians, ``ACTIVE_HAND_JOINT_NAMES`` order). This is a LINEAR
  PLACEHOLDER until the hardware bend ratio is measured (see
  ``JOINT_OPEN_POS_RAD`` / ``JOINT_CLOSED_POS_RAD``).

Pure numpy + math; importable without torch, mujoco or dex_retargeting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .igris_c_layout import ACTIVE_HAND_JOINT_NAMES, FIST_JOINT_POS_RAD

HAND_COMMAND_DIM = 12
HAND_KEYPOINT_COUNT = 26

# Wire order of HandCmd.motor_cmd: right 11..16, left 21..26. Within a hand:
# thumb flexion, index, middle, ring, little, thumb CMC (spread/opposition).
MOTOR_IDS = (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26)
HAND_SLOT_LABELS = ("thumb_flex", "index", "middle", "ring", "little", "thumb_cmc")
MOTOR_SLOT_LABELS = tuple(f"r_{label}" for label in HAND_SLOT_LABELS) + tuple(
    f"l_{label}" for label in HAND_SLOT_LABELS
)
RIGHT_MOTOR_SLOTS = slice(0, 6)
LEFT_MOTOR_SLOTS = slice(6, 12)
SIDE_MOTOR_SLOTS = {"right": RIGHT_MOTOR_SLOTS, "left": LEFT_MOTOR_SLOTS}

# Motor slot -> simulator joint. The thumb's two DoF sit in different positions
# in the motor block ([flex, ..., cmc]) and the layout block ([cmc, flex, ...]).
MOTOR_SLOT_TO_JOINT = (
    "r_1_joint_thumb_middle",
    "r_3_joint_index_middle",
    "r_5_joint_middle_middle",
    "r_7_joint_ring_middle",
    "r_9_joint_little_middle",
    "r_0_joint_thumb_proximal",
    "l_1_joint_thumb_middle",
    "l_3_joint_index_middle",
    "l_5_joint_middle_middle",
    "l_7_joint_ring_middle",
    "l_9_joint_little_middle",
    "l_0_joint_thumb_proximal",
)
MOTOR_ID_TO_JOINT = dict(zip(MOTOR_IDS, MOTOR_SLOT_TO_JOINT))
JOINT_TO_MOTOR_SLOT = {name: slot for slot, name in enumerate(MOTOR_SLOT_TO_JOINT)}

if len(MOTOR_IDS) != HAND_COMMAND_DIM or len(MOTOR_SLOT_TO_JOINT) != HAND_COMMAND_DIM:
    raise ImportError("Hand command layout must have exactly 12 motor slots.")
if set(MOTOR_SLOT_TO_JOINT) != set(ACTIVE_HAND_JOINT_NAMES) or len(set(MOTOR_SLOT_TO_JOINT)) != HAND_COMMAND_DIM:
    raise ImportError("MOTOR_SLOT_TO_JOINT must be a permutation of ACTIVE_HAND_JOINT_NAMES.")

# Simulator joint angle at command 0 (open) and 1 (closed), per active joint.
# Command 0 is the URDF zero for every joint: fingers straight and the thumb
# extended in the palm plane (the flat "paper" hand; operator-confirmed in sim
# on 2026-09-15 for the grip source). Command 1: FIST_JOINT_POS_RAD for the
# flexion joints; for the thumb CMC the URDF range limit (per-side sign,
# igris_c_v2_wrist_hand_isaac.urdf), i.e. the thumb rotated towards the palm.
# The same 0/1 direction was confirmed in the robot's MuJoCo model
# (igris_c_mujoco, 2026-09-15): q=0 extends every finger and the CMC, q=1
# moves the CMC towards the palm. Physical hand still to be checked.
CMC_CLOSED_POS_RAD = {
    "l_0_joint_thumb_proximal": 1.727875959,
    "r_0_joint_thumb_proximal": -1.727875959,
}
JOINT_OPEN_POS_RAD = {name: 0.0 for name in ACTIVE_HAND_JOINT_NAMES}
JOINT_CLOSED_POS_RAD = {name: CMC_CLOSED_POS_RAD.get(name, FIST_JOINT_POS_RAD[name]) for name in ACTIVE_HAND_JOINT_NAMES}

# OpenXR XR_EXT_hand_tracking joint indices: 0 Palm, 1 Wrist, 2-5 thumb
# (Metacarpal, Proximal, Distal, Tip), then per finger Metacarpal, Proximal,
# Intermediate, Distal, Tip for index (6-10), middle (11-15), ring (16-20),
# little (21-25). Each triple (a, b, c) contributes the angle at b between the
# bones a->b and b->c.
OPENXR_FLEXION_TRIPLES = {
    "thumb_flex": ((2, 3, 4), (3, 4, 5)),
    "index": ((6, 7, 8), (7, 8, 9), (8, 9, 10)),
    "middle": ((11, 12, 13), (12, 13, 14), (13, 14, 15)),
    "ring": ((16, 17, 18), (17, 18, 19), (18, 19, 20)),
    "little": ((21, 22, 23), (22, 23, 24), (23, 24, 25)),
}
# Thumb CMC: angle between the thumb metacarpal (2->3) and the index
# metacarpal (6->7). Which end of this angle is the extended in-plane thumb is
# a calibration question (see FingerCurlCalibration.cmc_*).
OPENXR_THUMB_CMC_BONES = ((2, 3), (6, 7))

# Controller input above this level counts as "the operator holds the controller".
CONTROLLER_ACTIVITY_THRESHOLD = 0.05

# Motor command at a fully squeezed controller grip, applied to all six motors of
# that hand (thumb CMC included). Chosen by the operator in the sim on 2026-09-15.
GRIP_FULL_COMMAND = 0.6

_DEGENERATE_BONE_M = 1.0e-6


@dataclass(frozen=True)
class FingerCurlCalibration:
    """Open/closed angle references in degrees for the normalization.

    ``*_open`` maps to command 0, ``*_closed`` to command 1; values outside are
    clamped. One set serves both hands (angles mirror). For the thumb CMC the
    small thumb-vs-index metacarpal angle is the open end: with this direction
    the operator's hand-tracked thumb reproduced the intended sim motion on
    2026-09-15 (the opposite direction moved the sim thumb the wrong way).

    TODO(human): defaults come from the right hand of
    hit-bench ``outputs/hand_recordings/pin_axes.vrs1`` (p05/p95 of the flexion
    sums); the left hand in that capture never closes fully. Re-derive from a
    two-hand open/fist capture (``inspect_hand_curl.py --propose``) and sign off.
    """

    thumb_flex_open_deg: float = 12.0
    thumb_flex_closed_deg: float = 55.0
    finger_open_deg: float = 15.0
    finger_closed_deg: float = 200.0
    cmc_open_deg: float = 30.0
    cmc_closed_deg: float = 70.0

    def __post_init__(self) -> None:
        for label in HAND_SLOT_LABELS:
            open_deg, closed_deg = self.bounds_deg(label)
            if not (math.isfinite(open_deg) and math.isfinite(closed_deg)):
                raise ValueError(f"Calibration for {label} must be finite.")
            if abs(closed_deg - open_deg) < 1.0e-3:
                raise ValueError(f"Calibration for {label} needs distinct open/closed angles.")

    def bounds_deg(self, label: str) -> tuple[float, float]:
        """(open, closed) degrees for one single-hand slot label."""

        if label == "thumb_flex":
            return self.thumb_flex_open_deg, self.thumb_flex_closed_deg
        if label == "thumb_cmc":
            return self.cmc_open_deg, self.cmc_closed_deg
        if label in ("index", "middle", "ring", "little"):
            return self.finger_open_deg, self.finger_closed_deg
        raise KeyError(label)


DEFAULT_CALIBRATION = FingerCurlCalibration()


def bone_angle(keypoints: np.ndarray, a: int, b: int, c: int) -> float:
    """Angle (rad) at ``b`` between the bones ``a->b`` and ``b->c``; 0 if degenerate."""

    u = keypoints[b] - keypoints[a]
    v = keypoints[c] - keypoints[b]
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u < _DEGENERATE_BONE_M or norm_v < _DEGENERATE_BONE_M:
        return 0.0
    cosine = float(np.dot(u, v)) / (norm_u * norm_v)
    return math.acos(max(-1.0, min(1.0, cosine)))


def finger_flexion_sums(keypoints: np.ndarray) -> np.ndarray:
    """Flexion angle sums (rad), order thumb, index, middle, ring, little."""

    return np.array(
        [
            sum(bone_angle(keypoints, *triple) for triple in OPENXR_FLEXION_TRIPLES[label])
            for label in HAND_SLOT_LABELS[:5]
        ],
        dtype=np.float64,
    )


def thumb_cmc_spread(keypoints: np.ndarray) -> float:
    """Angle (rad) between the thumb and index metacarpal bones; 0 if degenerate."""

    (a0, a1), (b0, b1) = OPENXR_THUMB_CMC_BONES
    u = keypoints[a1] - keypoints[a0]
    v = keypoints[b1] - keypoints[b0]
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u < _DEGENERATE_BONE_M or norm_v < _DEGENERATE_BONE_M:
        return 0.0
    cosine = float(np.dot(u, v)) / (norm_u * norm_v)
    return math.acos(max(-1.0, min(1.0, cosine)))


def normalize_angle(angle_rad: float, open_deg: float, closed_deg: float) -> float:
    """Linear map open->0, closed->1 of an angle, clamped to [0, 1]."""

    value = (math.degrees(angle_rad) - open_deg) / (closed_deg - open_deg)
    return min(1.0, max(0.0, value))


def hand_raw_angles(keypoints: np.ndarray) -> np.ndarray:
    """(6,) raw angles (rad) in ``HAND_SLOT_LABELS`` order, before normalization."""

    return np.concatenate((finger_flexion_sums(keypoints), [thumb_cmc_spread(keypoints)]))


def hand_keypoints_to_normalized(
    keypoints: Any,
    calibration: FingerCurlCalibration = DEFAULT_CALIBRATION,
) -> np.ndarray | None:
    """One hand's normalized command (6,) in ``HAND_SLOT_LABELS`` order.

    ``keypoints`` are the 26 wrist-relative OpenXR positions in meters, shape
    (26, 3). Returns ``None`` when the shape is wrong or any value is
    non-finite, so the caller holds the previous command for that hand.
    """

    array = np.asarray(keypoints, dtype=np.float64)
    if array.shape != (HAND_KEYPOINT_COUNT, 3) or not np.all(np.isfinite(array)):
        return None
    raw = hand_raw_angles(array)
    normalized = np.array(
        [normalize_angle(float(angle), *calibration.bounds_deg(label)) for angle, label in zip(raw, HAND_SLOT_LABELS)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(normalized)):
        return None
    return normalized


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def grip_to_normalized(
    left_grip: float,
    right_grip: float,
    *,
    full_command: float = GRIP_FULL_COMMAND,
) -> np.ndarray:
    """Controller grips onto the 12 motor slots (grips clamped; wire is unclamped).

    All six motors of a hand (thumb CMC included) follow that hand's grip
    linearly: released grip -> 0, fully squeezed grip -> ``full_command``.
    """

    command = np.zeros(HAND_COMMAND_DIM, dtype=np.float64)
    command[RIGHT_MOTOR_SLOTS] = _clamp01(right_grip) * float(full_command)
    command[LEFT_MOTOR_SLOTS] = _clamp01(left_grip) * float(full_command)
    return command


def _validate_command(command: Any) -> np.ndarray:
    array = np.asarray(command, dtype=np.float64).reshape(-1)
    if array.shape != (HAND_COMMAND_DIM,):
        raise ValueError(f"Hand command must have {HAND_COMMAND_DIM} values, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Hand command contains NaN or Inf.")
    return np.clip(array, 0.0, 1.0)


def normalized_to_joint_targets(command: Any) -> tuple[float, ...]:
    """12 motor-order commands -> radians in ``ACTIVE_HAND_JOINT_NAMES`` order.

    LINEAR PLACEHOLDER: ``JOINT_OPEN_POS_RAD`` (URDF zero) at 0 and
    ``JOINT_CLOSED_POS_RAD`` at 1 (fist target; thumb CMC: URDF limit, thumb
    towards the palm). Distal finger joints are not part of the command; the
    consumer couples them to the middle joints. Replace with a per-motor lookup
    table once the hardware bend ratio (joint angle vs. normalized target) has
    been measured.
    """

    values = _validate_command(command)
    targets = []
    for name in ACTIVE_HAND_JOINT_NAMES:
        open_pos = JOINT_OPEN_POS_RAD[name]
        closed_pos = JOINT_CLOSED_POS_RAD[name]
        fraction = float(values[JOINT_TO_MOTOR_SLOT[name]])
        targets.append(open_pos + fraction * (closed_pos - open_pos))
    return tuple(targets)


def motor_order_to_layout_order(command: Any) -> np.ndarray:
    """Reorder 12 motor-slot values into ``ACTIVE_HAND_JOINT_NAMES`` order."""

    array = np.asarray(command, dtype=np.float64).reshape(-1)
    if array.shape != (HAND_COMMAND_DIM,):
        raise ValueError(f"Expected {HAND_COMMAND_DIM} values, got shape {array.shape}.")
    return np.array([array[JOINT_TO_MOTOR_SLOT[name]] for name in ACTIVE_HAND_JOINT_NAMES], dtype=np.float64)


def layout_order_to_motor_order(values: Any) -> np.ndarray:
    """Reorder 12 ``ACTIVE_HAND_JOINT_NAMES``-ordered values into motor-slot order."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (HAND_COMMAND_DIM,):
        raise ValueError(f"Expected {HAND_COMMAND_DIM} values, got shape {array.shape}.")
    by_name = dict(zip(ACTIVE_HAND_JOINT_NAMES, array))
    return np.array([by_name[name] for name in MOTOR_SLOT_TO_JOINT], dtype=np.float64)


def controller_active(controller: Any, threshold: float = CONTROLLER_ACTIVITY_THRESHOLD) -> bool:
    """True when any analog/button channel of a ``ControllerInput`` exceeds ``threshold``."""

    if controller is None:
        return False
    for attribute in ("trigger", "grip", "button_first", "button_second", "button_thumbstick"):
        if abs(float(getattr(controller, attribute, 0.0))) > threshold:
            return True
    thumbstick = getattr(controller, "thumbstick", (0.0, 0.0))
    return any(abs(float(axis)) > threshold for axis in thumbstick)


HAND_SOURCES = ("hand", "grip", "hold")
FORCED_SOURCES = (None, "hand", "grip")


class HandSourceArbiter:
    """Per-hand choice between hand tracking, controller grip, and holding.

    The VRS1 uplink carries no explicit "controller held" flag; the headset's
    own swap signal is the per-frame ``HandState.tracked``. Rules per hand:

    * forced ``"grip"`` -> ``"grip"``; forced ``"hand"`` -> ``"hand"`` while
      tracked, otherwise ``"hold"``.
    * auto: no hand state at all (sender without hand tracking) -> ``"grip"``;
      tracked -> ``"hand"``; untracked with controller activity -> ``"grip"``
      (the operator picked the controller up); untracked and idle -> ``"hold"``
      (an occluded bare hand must not open and drop what it holds).

    ``"grip"`` writes ``grip * GRIP_FULL_COMMAND`` to all six motors of the hand.

    Stateless per frame; the last decision per side is exposed for reporting.
    """

    def __init__(self, forced: str | None = None, *, activity_threshold: float = CONTROLLER_ACTIVITY_THRESHOLD) -> None:
        if forced not in FORCED_SOURCES:
            raise ValueError(f"forced must be one of {FORCED_SOURCES}, got {forced!r}.")
        self._forced = forced
        self._threshold = float(activity_threshold)
        self._active = {"left": "hold", "right": "hold"}

    @property
    def forced(self) -> str | None:
        return self._forced

    @property
    def active_sources(self) -> tuple[str, str]:
        """(left, right) decision of the most recent ``decide`` per side."""

        return self._active["left"], self._active["right"]

    def reset(self) -> None:
        self._active = {"left": "hold", "right": "hold"}

    def decide(self, side: str, hand_state: Any, controller_input: Any) -> str:
        if side not in self._active:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
        tracked = hand_state is not None and bool(getattr(hand_state, "tracked", False))
        if self._forced == "grip":
            decision = "grip"
        elif self._forced == "hand":
            decision = "hand" if tracked else "hold"
        elif hand_state is None:
            decision = "grip"
        elif tracked:
            decision = "hand"
        elif controller_active(controller_input, self._threshold):
            decision = "grip"
        else:
            decision = "hold"
        self._active[side] = decision
        return decision


class HandCommandState:
    """Owns the 12-motor command and fills it per hand from the arbiter's decision.

    ``update`` returns the effective decision per side after degradation:
    ``"hand"`` when fresh keypoints produced the block, ``"grip"`` when the
    controller grip did, ``"hold"`` when the block was left untouched
    (arbiter hold, or hand tracking selected but the keypoints were unusable).
    The command persists across ``reset`` so a held hand keeps its last value
    downstream as well.
    """

    def __init__(
        self,
        forced: str | None = None,
        *,
        calibration: FingerCurlCalibration = DEFAULT_CALIBRATION,
        curl_fn=hand_keypoints_to_normalized,
        activity_threshold: float = CONTROLLER_ACTIVITY_THRESHOLD,
    ) -> None:
        self._arbiter = HandSourceArbiter(forced, activity_threshold=activity_threshold)
        self._calibration = calibration
        self._curl_fn = curl_fn
        self._command = np.zeros(HAND_COMMAND_DIM, dtype=np.float64)
        self._decisions = {"left": "hold", "right": "hold"}

    @property
    def command(self) -> np.ndarray:
        """Copy of the current normalized command (``MOTOR_IDS`` order)."""

        return self._command.copy()

    @property
    def active_sources(self) -> tuple[str, str]:
        """(left, right) effective source of the latest ``update``."""

        return self._decisions["left"], self._decisions["right"]

    @property
    def fresh_tracking(self) -> tuple[bool, bool]:
        """(left, right): whether the latest ``update`` used fresh hand-tracking keypoints."""

        return self._decisions["left"] == "hand", self._decisions["right"] == "hand"

    def reset(self) -> None:
        """Stream lost: forget the decisions, keep the command (hold)."""

        self._arbiter.reset()
        self._decisions = {"left": "hold", "right": "hold"}

    def update(self, left_hand: Any, right_hand: Any, left_controller: Any, right_controller: Any) -> dict[str, str]:
        for side, hand, controller in (("left", left_hand, left_controller), ("right", right_hand, right_controller)):
            decision = self._arbiter.decide(side, hand, controller)
            block = SIDE_MOTOR_SLOTS[side]
            if decision == "hand":
                normalized = None
                keypoints = getattr(hand, "keypoints", ())
                if len(keypoints) == HAND_KEYPOINT_COUNT:
                    positions = [tuple(keypoint.position_m) for keypoint in keypoints]
                    normalized = self._curl_fn(np.asarray(positions, dtype=np.float64), self._calibration)
                if normalized is None:
                    decision = "hold"
                else:
                    self._command[block] = np.asarray(normalized, dtype=np.float64).reshape(-1)
            elif decision == "grip":
                self._command[block] = _clamp01(getattr(controller, "grip", 0.0)) * GRIP_FULL_COMMAND
            self._decisions[side] = decision
        return dict(self._decisions)


__all__ = [
    "CMC_CLOSED_POS_RAD",
    "HandCommandState",
    "JOINT_CLOSED_POS_RAD",
    "JOINT_OPEN_POS_RAD",
    "CONTROLLER_ACTIVITY_THRESHOLD",
    "DEFAULT_CALIBRATION",
    "FORCED_SOURCES",
    "GRIP_FULL_COMMAND",
    "FingerCurlCalibration",
    "HAND_COMMAND_DIM",
    "HAND_KEYPOINT_COUNT",
    "HAND_SLOT_LABELS",
    "HAND_SOURCES",
    "HandSourceArbiter",
    "JOINT_TO_MOTOR_SLOT",
    "LEFT_MOTOR_SLOTS",
    "MOTOR_IDS",
    "MOTOR_ID_TO_JOINT",
    "MOTOR_SLOT_LABELS",
    "MOTOR_SLOT_TO_JOINT",
    "OPENXR_FLEXION_TRIPLES",
    "OPENXR_THUMB_CMC_BONES",
    "RIGHT_MOTOR_SLOTS",
    "SIDE_MOTOR_SLOTS",
    "bone_angle",
    "controller_active",
    "finger_flexion_sums",
    "grip_to_normalized",
    "hand_keypoints_to_normalized",
    "hand_raw_angles",
    "layout_order_to_motor_order",
    "motor_order_to_layout_order",
    "normalize_angle",
    "normalized_to_joint_targets",
    "thumb_cmc_spread",
]
