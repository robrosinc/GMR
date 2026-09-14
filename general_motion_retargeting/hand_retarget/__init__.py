"""IGRIS-C hand (finger) retargeting side channel.

Only the layout constants are re-exported here so that importing this package
never loads torch/dex_retargeting; import the solver explicitly:

    from general_motion_retargeting.hand_retarget.igris_c_hand import IgrisHandRetargeter
"""

from .igris_c_layout import (
    ACTIVE_HAND_JOINT_NAMES,
    AUXILIARY_JOINT_NAMES,
    DISTAL_COUPLING,
    FIST_JOINT_POS_RAD,
    LEFT_HAND_JOINT_NAMES,
    NECK_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
    WRIST_JOINT_NAMES,
    grip_hand_joint_targets,
)

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
