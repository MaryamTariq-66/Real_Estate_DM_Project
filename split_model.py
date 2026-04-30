from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SOURCE = Path("property_model.pkl")
DEFAULT_PARTS_DIR = Path("property_model.pkl.parts")
DEFAULT_CHUNK_SIZE = 24 * 1024 * 1024


def split_file(source_path: Path, parts_dir: Path, chunk_size: int) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Source model file not found: {source_path}")

    parts_dir.mkdir(parents=True, exist_ok=True)

    for existing_part in parts_dir.glob(f"{source_path.name}.part*"):
        existing_part.unlink()

    part_index = 0
    with source_path.open("rb") as source_file:
        while True:
            chunk = source_file.read(chunk_size)
            if not chunk:
                break

            part_path = parts_dir / f"{source_path.name}.part{part_index:03d}"
            with part_path.open("wb") as part_file:
                part_file.write(chunk)
            part_index += 1

    if part_index == 0:
        raise ValueError(f"Source file is empty: {source_path}")

    print(f"Created {part_index} chunk files in {parts_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the trained model into smaller chunks.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to the full model pickle")
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR, help="Directory for chunk files")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in bytes. Default keeps parts under 25 MB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_file(args.source, args.parts_dir, args.chunk_size)


if __name__ == "__main__":
    main()
