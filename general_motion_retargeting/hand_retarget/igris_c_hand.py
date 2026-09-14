"""DexPilot fingertip retargeting from OpenXR hand keypoints to IGRIS-C fingers.

Solves "26 wrist-relative keypoint positions -> 6 actuated finger joint angles
per hand" with dex-retargeting's DexPilot optimizer, using the retargeting-only
hand URDFs and YAML configs under ``assets/robros_igris_c_v2/hand_retarget/``
(no meshes; kinematics only). The distal joints are NOT solved here: they
follow the actuated middle joints via ``DISTAL_COUPLING`` applied by the
consumer (the retarget URDFs mirror that coupling with ``<mimic>`` tags so the
optimizer's fingertip FK stays consistent).

Input contract matches the VRS1 ``HandState.keypoints`` of the vr_link_schema
uplink: 26 OpenXR ``XR_EXT_hand_tracking`` joints, positions in meters,
wrist-relative in the wrist frame (entry 1 = wrist = identity). Output is
radians in ``ACTIVE_HAND_JOINT_NAMES`` order (6 per hand). Conversion to the
robot hand's normalized 0..1 motor targets is the consumer's job.

Moved from hit-bench ``source/command_source/motion_gen/vr/hand_retargeter.py``
(2026-09-14) unchanged apart from the asset path and imports. dex_retargeting
and torch are imported lazily in ``__init__`` so importing GMR never requires
them; install with ``pip install -e .[hand]`` (dex-retargeting pins numpy<2).

Known open tuning items (human sign-off pending, carried over from hit-bench):
``scaling_factor``/``low_pass_alpha`` in the YAMLs, the 0.04 m fingertip
offset in the URDFs, and the thumb spread axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..params import ASSET_ROOT
from .igris_c_layout import LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES

# OpenXR XR_EXT_hand_tracking (26) -> MANO (21) row selection: drops Palm (0)
# and the four non-thumb metacarpals (6, 11, 16, 21). Matches IsaacLab's
# _HAND_JOINTS_INDEX in g1_dex_retargeting_utils.py.
OPENXR_TO_MANO_INDEX = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25)

# Wrist-frame -> retarget-frame rotation (the IGRIS palm frame: fingers +z,
# index side +x, right palm facing -y / left +y). Pinned empirically against
# the Unreal sender on the hit-bench recording
# outputs/hand_recordings/pin_axes.vrs1 (2026-08-19) with
# scripts/teleoperation/replay_hand_retarget.py: --search top proper-rotation
# (det=+1) candidate per side, confirmed by the fist-curl direction matching
# the robot FK (right fist tips y<0, left y>0) and by open-hand angle traces
# sitting near 0 rad. The det=-1 twins of these score identically on the
# geometric heuristic but mirror the hand - reject them.
DEFAULT_OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)
DEFAULT_OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ],
    dtype=np.float64,
)

DEFAULT_CONFIG_DIR = Path(ASSET_ROOT).resolve() / "robros_igris_c_v2" / "hand_retarget"

HAND_KEYPOINT_COUNT = 26

_SIDES = ("left", "right")


class IgrisHandRetargeter:
    """Per-frame OpenXR-keypoints -> IGRIS-C finger joint retargeting."""

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        operator2mano_left: np.ndarray = DEFAULT_OPERATOR2MANO_LEFT,
        operator2mano_right: np.ndarray = DEFAULT_OPERATOR2MANO_RIGHT,
    ) -> None:
        import torch
        from dex_retargeting.retargeting_config import RetargetingConfig

        self._torch = torch
        self._operator2mano = {
            "left": np.asarray(operator2mano_left, dtype=np.float64),
            "right": np.asarray(operator2mano_right, dtype=np.float64),
        }

        config_dir = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
        self._retargetings = {}
        self._output_indices = {}
        expected_names = {
            "left": tuple(LEFT_HAND_JOINT_NAMES),
            "right": tuple(RIGHT_HAND_JOINT_NAMES),
        }
        for side in _SIDES:
            yaml_path = config_dir / f"dexpilot_{side}.yml"
            urdf_path = config_dir / f"igris_c_hand_{side}_retarget.urdf"
            if not yaml_path.is_file() or not urdf_path.is_file():
                raise FileNotFoundError(
                    f"Hand retarget assets for the {side} hand not found in {config_dir} "
                    f"(expected {yaml_path.name} and {urdf_path.name})."
                )
            # Resolve the URDF next to the YAML explicitly instead of relying
            # on the library's process-global default URDF directory.
            config = RetargetingConfig.load_from_file(str(yaml_path), override={"urdf_path": str(urdf_path)})
            retargeting = config.build()
            self._retargetings[side] = retargeting

            # retarget() returns the full dof vector (actuated + mimic) in
            # robot.dof_joint_names order; map it to the auxiliary-channel
            # order by name and fail loudly on any mismatch.
            dof_names = retargeting.optimizer.robot.dof_joint_names
            missing = [name for name in expected_names[side] if name not in dof_names]
            if missing:
                raise ValueError(
                    f"Retarget URDF for the {side} hand is missing joints {missing}; got {dof_names}."
                )
            self._output_indices[side] = np.array(
                [dof_names.index(name) for name in expected_names[side]], dtype=np.int64
            )

    @property
    def left_joint_names(self) -> tuple[str, ...]:
        return tuple(LEFT_HAND_JOINT_NAMES)

    @property
    def right_joint_names(self) -> tuple[str, ...]:
        return tuple(RIGHT_HAND_JOINT_NAMES)

    def solve_hand(self, side: str, keypoints_wrist: np.ndarray) -> np.ndarray:
        """Retarget one hand's keypoints to its 6 actuated joint angles.

        Args:
            side: "left" or "right".
            keypoints_wrist: (26, 3) keypoint positions in meters, wrist
                relative in the wrist frame, OpenXR joint order.

        Returns:
            (6,) joint angles in radians, ordered like the corresponding
            ``ACTIVE_HAND_JOINT_NAMES`` slice (limits enforced by the
            optimizer).
        """

        if side not in _SIDES:
            raise ValueError(f"Unknown hand side {side!r}.")
        keypoints_wrist = np.asarray(keypoints_wrist, dtype=np.float64)
        if keypoints_wrist.shape != (HAND_KEYPOINT_COUNT, 3):
            raise ValueError(f"Expected ({HAND_KEYPOINT_COUNT}, 3) keypoints, got {keypoints_wrist.shape}.")

        mano_keypoints = keypoints_wrist[list(OPENXR_TO_MANO_INDEX)]
        # Keypoints are already wrist-relative (entry 1 is identity), so only
        # re-zero on the wrist row for robustness and rotate into MANO axes;
        # the world->wrist step of the IsaacLab template does not apply here.
        mano_keypoints = (mano_keypoints - mano_keypoints[0:1, :]) @ self._operator2mano[side]

        retargeting = self._retargetings[side]
        indices = retargeting.optimizer.target_link_human_indices
        ref_value = mano_keypoints[indices[1, :], :] - mano_keypoints[indices[0, :], :]

        # The optimizer differentiates through torch; some host processes
        # (Isaac runners) disable grad/inference modes globally.
        with self._torch.enable_grad(), self._torch.inference_mode(False):
            qpos_full = retargeting.retarget(ref_value)

        return np.asarray(qpos_full, dtype=np.float64)[self._output_indices[side]]

    def solve(
        self,
        left_keypoints: np.ndarray | None,
        right_keypoints: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Retarget both hands; a ``None`` input yields a ``None`` output."""

        left = None if left_keypoints is None else self.solve_hand("left", left_keypoints)
        right = None if right_keypoints is None else self.solve_hand("right", right_keypoints)
        return left, right

    def reset(self, side: str | None = None) -> None:
        """Drop warm-start and low-pass state (call on tracking reacquisition).

        Without this, the optimizer seed and output filter would drag the
        stale pre-loss pose into the first frames after a hand is reacquired.
        """

        sides = _SIDES if side is None else (side,)
        for name in sides:
            retargeting = self._retargetings[name]
            retargeting.reset()
            if retargeting.filter is not None:
                retargeting.filter.reset()


__all__ = [
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_OPERATOR2MANO_LEFT",
    "DEFAULT_OPERATOR2MANO_RIGHT",
    "HAND_KEYPOINT_COUNT",
    "IgrisHandRetargeter",
    "OPENXR_TO_MANO_INDEX",
]
