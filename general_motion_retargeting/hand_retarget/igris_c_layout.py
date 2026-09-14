"""Fixed auxiliary-joint layout of the Robros IGRIS-C v2 (wrists, fingers, neck).

Single source for the joint names that travel beside GMR's 31-dof body
retargeting: the producer publishes finger targets in ``ACTIVE_HAND_JOINT_NAMES``
order, hit-bench's Isaac auxiliary controller consumes the same order, and the
distal finger joints follow their actuated middle joint through
``DISTAL_COUPLING``. Moved here from hit-bench
``source/embodiments/igris_c/controllers/auxiliary_joint_layout.py`` (values
unchanged) so that the hand retargeter and its consumers share one definition.
"""

from __future__ import annotations

import math

NECK_JOINT_NAMES = (
    "Joint_Neck_Yaw",
    "Joint_Neck_Pitch",
)

WRIST_JOINT_NAMES = (
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
)

# Driven finger joints (6 per hand, left then right). ``*_0_joint_thumb_proximal``
# is the thumb abduction/spread axis (right hand: negative range), the others are
# flexion of the middle phalanx; distal joints are coupled, see DISTAL_COUPLING.
ACTIVE_HAND_JOINT_NAMES = (
    "l_0_joint_thumb_proximal",
    "l_1_joint_thumb_middle",
    "l_3_joint_index_middle",
    "l_5_joint_middle_middle",
    "l_7_joint_ring_middle",
    "l_9_joint_little_middle",
    "r_0_joint_thumb_proximal",
    "r_1_joint_thumb_middle",
    "r_3_joint_index_middle",
    "r_5_joint_middle_middle",
    "r_7_joint_ring_middle",
    "r_9_joint_little_middle",
)

LEFT_HAND_JOINT_NAMES = ACTIVE_HAND_JOINT_NAMES[:6]
RIGHT_HAND_JOINT_NAMES = ACTIVE_HAND_JOINT_NAMES[6:]

AUXILIARY_JOINT_NAMES = WRIST_JOINT_NAMES + ACTIVE_HAND_JOINT_NAMES + NECK_JOINT_NAMES

# (finger, actuated middle joint, coupled distal joint, multiplier):
# distal_target = middle_target * multiplier.
DISTAL_COUPLING = (
    ("l_thumb", "l_1_joint_thumb_middle", "l_2_joint_thumb_distal", 1.285714286),
    ("l_index", "l_3_joint_index_middle", "l_4_joint_index_distal", 1.0),
    ("l_middle", "l_5_joint_middle_middle", "l_6_joint_middle_distal", 1.0),
    ("l_ring", "l_7_joint_ring_middle", "l_8_joint_ring_distal", 1.0),
    ("l_little", "l_9_joint_little_middle", "l_10_joint_little_distal", 1.0),
    ("r_thumb", "r_1_joint_thumb_middle", "r_2_joint_thumb_distal", 1.285714286),
    ("r_index", "r_3_joint_index_middle", "r_4_joint_index_distal", 1.0),
    ("r_middle", "r_5_joint_middle_middle", "r_6_joint_middle_distal", 1.0),
    ("r_ring", "r_7_joint_ring_middle", "r_8_joint_ring_distal", 1.0),
    ("r_little", "r_9_joint_little_middle", "r_10_joint_little_distal", 1.0),
)

# Fully closed ("fist") targets in radians; the open hand is all zeros. Values
# follow the only sim-exercised closed-hand pose (hit-bench
# scripts/tools/create_right_arm_finger_motion.py); the thumb is untuned and
# expects human adjustment on the robot to avoid self-intersection.
FIST_JOINT_POS_RAD: dict[str, float] = {
    name: (
        0.0
        if name.endswith("_0_joint_thumb_proximal")
        else math.radians(70.0)
        if name.endswith("_1_joint_thumb_middle")
        else math.pi / 2.0
    )
    for name in ACTIVE_HAND_JOINT_NAMES
}


def grip_hand_joint_targets(left_grip: float, right_grip: float) -> tuple[float, ...]:
    """Open-to-fist finger targets scaled by the (clamped) controller grips.

    Returns radians in ``ACTIVE_HAND_JOINT_NAMES`` order; grips are clamped to
    [0, 1] since the VR wire carries them unclamped.
    """

    left = min(1.0, max(0.0, float(left_grip)))
    right = min(1.0, max(0.0, float(right_grip)))
    return tuple((left if name.startswith("l_") else right) * FIST_JOINT_POS_RAD[name] for name in ACTIVE_HAND_JOINT_NAMES)


__all__ = [
    "ACTIVE_HAND_JOINT_NAMES",
    "AUXILIARY_JOINT_NAMES",
    "DISTAL_COUPLING",
    "FIST_JOINT_POS_RAD",
    "LEFT_HAND_JOINT_NAMES",
    "NECK_JOINT_NAMES",
    "RIGHT_HAND_JOINT_NAMES",
    "WRIST_JOINT_NAMES",
    "grip_hand_joint_targets",
]
