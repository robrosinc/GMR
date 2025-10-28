import argparse
import pathlib
import pickle
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

PHC_REQUIRED_KEYS = ("root_trans_offset", "root_rot", "dof", "fps")


def _is_phc_motion(candidate: Any) -> bool:
    return isinstance(candidate, dict) and all(key in candidate for key in PHC_REQUIRED_KEYS)


def _extract_phc_motion(payload: Any) -> Tuple[Dict[str, Any], Callable[[Dict[str, Any]], Dict[str, Any]]]:
    if _is_phc_motion(payload):
        return payload, lambda phc_motion: phc_motion

    if isinstance(payload, dict) and "motion" in payload and _is_phc_motion(payload["motion"]):
        items = list(payload.items())
        motion_index = next(i for i, (key, _) in enumerate(items) if key == "motion")
        container_type = payload.__class__

        def wrap(phc_motion: Dict[str, Any]) -> Dict[str, Any]:
            updated_items = []
            for idx, (key, value) in enumerate(items):
                if idx == motion_index:
                    updated_items.append(("motion", phc_motion))
                else:
                    updated_items.append((key, value))
            return container_type(updated_items)

        return payload["motion"], wrap

    raise ValueError("Input pickle must contain a PHC motion dict either directly or under the 'motion' key.")


def convert_phc_to_gmr(phc_motion: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single-motion PHC dict to GMR dict.

    Required input keys (PHC): 'root_trans_offset', 'root_rot', 'dof', 'fps'
    Output keys (GMR): 'root_pos', 'root_rot', 'dof_pos', 'fps'

    The function copies arrays by reference (no deep copy) to be lightweight.
    """

    if not isinstance(phc_motion, dict):
        raise TypeError("phc_motion must be a dict")

    missing = [k for k in ("root_trans_offset", "root_rot", "dof", "fps") if k not in phc_motion]
    if missing:
        raise KeyError(f"PHC motion missing required keys: {missing}")

    root_trans_offset = phc_motion["root_trans_offset"]
    root_rot = phc_motion["root_rot"]
    dof = phc_motion["dof"]
    dof_vel = phc_motion["dof_vel"]
    root_vel = phc_motion["root_vel"]
    root_angvel = phc_motion["root_angvel"]
    kb_pos = phc_motion["keybody_pos"]
    fps = phc_motion["fps"]

    # Basic shape checks (best-effort)
    if not isinstance(root_trans_offset, np.ndarray) or root_trans_offset.ndim != 2 or root_trans_offset.shape[1] != 3:
        raise ValueError("root_trans_offset must be ndarray of shape [T, 3]")
    if not isinstance(root_rot, np.ndarray) or root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError("root_rot must be ndarray of shape [T, 4]")
    if not isinstance(dof, np.ndarray) or dof.ndim != 2:
        raise ValueError("dof must be ndarray of shape [T, D]")
    if root_trans_offset.shape[0] != root_rot.shape[0] or root_trans_offset.shape[0] != dof.shape[0]:
        raise ValueError("All arrays must have the same time dimension T")

    gmr_motion: Dict[str, Any] = {
        "fps": fps,
        "root_pos": root_trans_offset,
        "root_rot": root_rot,
        "dof_pos": dof,
        'dof_vel': dof_vel,
        'root_vel': root_vel,
        'root_angvel': root_angvel,
        'kb_pos': kb_pos,
        'local_body_pos': None,
        'link_body_list': None,
    }
    print(dof_vel)

    return gmr_motion


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a GMR pickle file to PHC format.")
    parser.add_argument("input_path", type=pathlib.Path, help="Path to the source GMR pickle file.")
    parser.add_argument(
        "output_path",
        type=pathlib.Path,
        nargs="?",
        help="Optional target path for the PHC pickle (defaults to <input>_phc.pkl).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )

    args = parser.parse_args(argv)

    input_path = args.input_path.expanduser()
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    output_path = args.output_path.expanduser() if args.output_path else input_path.with_name(f"{input_path.stem}_phc.pkl")

    if output_path.exists() and not args.overwrite:
        parser.error(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    with open(input_path, "rb") as src:
        payload = pickle.load(src)

    try:
        gmr_motion, wrap_payload = _extract_phc_motion(payload)
    except ValueError as exc:
        parser.error(str(exc))

    phc_motion = convert_phc_to_gmr(gmr_motion)
    output_payload = wrap_payload(phc_motion)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as dst:
        pickle.dump(output_payload, dst)

    print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
