
import pickle
import joblib
from pathlib import Path

def safe_load_pickle(path: str | Path):
    """Load pickles robustly across numpy/pickle versions."""
    path = Path(path)
    try:
        return joblib.load(path)
    except Exception as exc:
        try:
            with path.open("rb") as handle:
                class _NumpyCompatUnpickler(pickle.Unpickler):
                    _module_map = {
                        "numpy._core": "numpy.core",
                        "numpy._core.multiarray": "numpy.core.multiarray",
                        "numpy._core.numeric": "numpy.core.numeric",
                        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
                    }

                    def find_class(self, module, name):
                        module = self._module_map.get(module, module)
                        if module.startswith("numpy._core."):
                            module = "numpy.core." + module.split("numpy._core.", 1)[1]
                        return super().find_class(module, name)

                return _NumpyCompatUnpickler(handle).load()
        except Exception:
            raise exc

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = safe_load_pickle(motion_file) #pickle.load(f)
        motion_fps = motion_data["fps"]
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_dof_pos = motion_data["dof_pos"]
        motion_local_body_pos = motion_data["local_body_pos"]
        try:
            motion_link_body_list = motion_data["link_body_list"]
        except:
            motion_link_body_list = None
            
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list


