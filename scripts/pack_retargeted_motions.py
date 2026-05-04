import argparse
import pickle
from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError:
    joblib = None


def safe_load_pickle(path: Path) -> Any:
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass

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


def build_clip_name(base_dir: Path, file_path: Path) -> str:
    relative = file_path.relative_to(base_dir).with_suffix("")
    return relative.as_posix()


def collect_pkl_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.pkl") if path.is_file())


def resolve_output_path(input_dir: Path, output_name: str) -> Path:
    output_path = Path(output_name).expanduser()
    if output_path.is_absolute():
        return output_path
    return input_dir / output_path


def is_single_motion_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    required_keys = {"fps", "root_pos", "root_rot", "dof_pos"}
    return required_keys.issubset(obj.keys())


def _join_clip_name(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}/{suffix}"


def flatten_motion_payload(payload: Any, base_clip_name: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _walk(node: Any, clip_name: str) -> None:
        if is_single_motion_dict(node):
            if clip_name in flat:
                raise ValueError(f"Duplicate clip name detected while flattening: {clip_name}")
            flat[clip_name] = node
            return

        if isinstance(node, dict):
            if not node:
                if clip_name in flat:
                    raise ValueError(f"Duplicate clip name detected while flattening: {clip_name}")
                flat[clip_name] = node
                return
            for sub_key, sub_val in node.items():
                _walk(sub_val, _join_clip_name(clip_name, str(sub_key)))
            return

        if clip_name in flat:
            raise ValueError(f"Duplicate clip name detected while flattening: {clip_name}")
        flat[clip_name] = node

    _walk(payload, base_clip_name)
    return flat


def pack_motions(input_dir: Path, output_path: Path) -> tuple[int, Path]:
    pkl_files = collect_pkl_files(input_dir)
    output_abs = output_path.resolve()
    packed: dict[str, Any] = {}

    for file_path in pkl_files:
        if file_path.resolve() == output_abs:
            continue
        clip_name = build_clip_name(input_dir, file_path)
        payload = safe_load_pickle(file_path)
        flat_payload = flatten_motion_payload(payload, clip_name)
        for flat_clip_name, flat_clip_data in flat_payload.items():
            if flat_clip_name in packed:
                raise ValueError(f"Duplicate clip name detected across files: {flat_clip_name}")
            packed[flat_clip_name] = flat_clip_data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(packed, f, protocol=pickle.HIGHEST_PROTOCOL)

    return len(packed), output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack all .pkl files under a directory into a single dict pickle."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to scan recursively for .pkl files.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="motion.pkl",
        help="Output pickle filename or path. Default: motion.pkl",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_path = resolve_output_path(input_dir, args.output_name)
    clip_count, saved_path = pack_motions(input_dir, output_path)

    print(f"Packed {clip_count} clips -> {saved_path}")


if __name__ == "__main__":
    main()
