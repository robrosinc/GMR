import numpy as np
import mujoco as mj

from .motion_retarget import GeneralMotionRetargeting
from .params import ROBOT_XML_DICT


class RobotToRobotMotionRetargeting:
    """Retarget a saved source robot motion through the existing GMR IK solver."""

    def __init__(
        self,
        src_robot: str,
        tgt_robot: str,
        solver: str = "daqp",
        damping: float = 5e-1,
        verbose: bool = True,
        use_velocity_limit: bool = False,
    ) -> None:
        self.src_robot = src_robot
        self.tgt_robot = tgt_robot

        self.source_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[src_robot]))
        self.source_data = mj.MjData(self.source_model)

        self.retargeter = GeneralMotionRetargeting(
            src_human=src_robot,
            tgt_robot=tgt_robot,
            solver=solver,
            damping=damping,
            verbose=verbose,
            use_velocity_limit=use_velocity_limit,
        )

        self.source_body_names = tuple(self.retargeter.human_scale_table.keys())
        self.source_body_ids = self._resolve_source_body_ids(self.source_body_names)

    @property
    def model(self):
        return self.retargeter.model

    @property
    def robot_body_names(self):
        return self.retargeter.robot_body_names

    @property
    def scaled_source_data(self):
        return self.retargeter.scaled_human_data

    def source_frame_from_qpos(self, qpos: np.ndarray) -> dict[str, list[np.ndarray]]:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.source_model.nq,):
            raise ValueError(
                f"{self.src_robot} qpos must have shape {(self.source_model.nq,)}, "
                f"got {qpos.shape}."
            )

        self.source_data.qpos[:] = qpos
        mj.mj_forward(self.source_model, self.source_data)

        return {
            body_name: [
                self.source_data.xpos[body_id].copy(),
                self.source_data.xquat[body_id].copy(),
            ]
            for body_name, body_id in self.source_body_ids.items()
        }

    def retarget(self, source_qpos: np.ndarray, offset_to_ground: bool = False) -> np.ndarray:
        source_frame = self.source_frame_from_qpos(source_qpos)
        return self.retargeter.retarget(source_frame, offset_to_ground=offset_to_ground)

    def _resolve_source_body_ids(self, body_names: tuple[str, ...]) -> dict[str, int]:
        body_ids = {}
        missing_body_names = []

        for body_name in body_names:
            body_id = mj.mj_name2id(self.source_model, mj.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                missing_body_names.append(body_name)
            else:
                body_ids[body_name] = body_id

        if missing_body_names:
            raise ValueError(
                f"{self.src_robot} model is missing bodies required by the IK config: "
                f"{missing_body_names}"
            )

        return body_ids
