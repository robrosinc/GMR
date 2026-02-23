import argparse
import pickle
import numpy as np
import joblib
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> None:
    """Convert npz motion data to gmr pickle format."""
    data = np.load(input_path)
    clean = {k: data[k] for k in data.files}

    with output_path.open("wb") as f:
        clean["fps"] = 30
        final_data = clean
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    arr = joblib.load(output_path)
    arr.keys()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert npz motion data to GMR pickle format."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input npz file (e.g., test.npz)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("robot_motion.pkl"),
        help="Output pickle file (default: robot_motion.pkl)",
    )
    args = parser.parse_args()

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
