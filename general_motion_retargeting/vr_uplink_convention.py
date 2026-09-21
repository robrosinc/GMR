"""Joint-orientation convention adapter for the VR (VRS1) pose uplink.

Input-side counterpart of the action adapter: it converts the SENDER's joint
orientation convention into the one GMR's ``xrobot_to_igris_c.json`` rot_offset
calibration expects, BEFORE retargeting. It never touches positions and has
nothing to do with policy actions.

It is consumed through the ``body_joint_adapter`` hook of the ``vr_link``
receiver (vr_link_schema repository): ``to_xrobot_convention(joint_name,
quat_wxyz) -> quat_wxyz`` is exactly that hook's signature, so a consumer
passes the function to ``vr_link.TcpVRDevice`` / ``decode_vr_state_payload``.
This module deliberately imports nothing from ``vr_link``: the constant
belongs to the IK configuration it was solved against, not to the wire
contract.

Sender convention (verified from live captures 2026-07-28): world-aligned bone
frames with T-pose neutral (SMPL-like), proper right-handed Z-up world
quaternions in wxyz.

The conversion is a single constant right-multiplication:

    q_gmr = q_sender x (0.5, 0.5, -0.5, -0.5)

It was SOLVED, not guessed: the zero-IK-cost human quaternion for each config
row is ``FK_link_quat(robot pose) x rot_offset^-1``; solving
``A x q_sender x B = q_required`` over matching robot/human poses (arms down,
arms out, yawed 30 deg) gave A = identity and this B for every joint with 0.0
deg residual (scratchpad solve_conversion.py). Full-pipeline verification:
upright 0.6 deg, hands correctly below pelvis, T-pose arm spread 1.36 m, yaw
error 0.0 deg - the only candidate satisfying all metrics at once. Earlier
single-sided Rx90 variants were ~90 deg off, which the position-weighted IK
hid on the torso (slight forward lean) but exposed on the orientation-only
arm tasks (raised arm).

Per-joint entries exist so residual offsets could be tuned individually, but
the solved constant fits all joints; keep the table uniform unless hardware
testing shows a joint-specific residual.
"""

from __future__ import annotations

from .raw_xrobot_motion import XROBOT_BODY_JOINT_NAMES

Quat = tuple[float, float, float, float]

# Solved conversion constant: maps sender bone frames onto the convention the
# IK config's rot_offsets were authored against (robot qpos0 <-> this value).
_SOLVED_B_WXYZ: Quat = (0.5, 0.5, -0.5, -0.5)

JOINT_CORRECTIONS_WXYZ: dict[str, Quat] = {
    name: _SOLVED_B_WXYZ for name in XROBOT_BODY_JOINT_NAMES
}


def quat_mul_wxyz(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def to_xrobot_convention(joint_name: str, quat_wxyz: Quat) -> Quat:
    """Convert one sender-convention joint orientation to the GMR convention."""

    return quat_mul_wxyz(quat_wxyz, JOINT_CORRECTIONS_WXYZ[joint_name])
