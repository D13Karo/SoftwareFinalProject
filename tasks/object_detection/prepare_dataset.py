"""Convert a folder of labelme JSON annotations into YOLO-format labels.

Walks a directory tree, finds every ``*.json`` produced by labelme, and writes
a matching ``*.txt`` file next to it (or into --out-dir if specified). The
output format is the YOLO label-text format that the Colab training notebook
expects: one line per box, ``<class_id> <cx> <cy> <w> <h>`` with all values
normalized to [0, 1].

Image dimensions are read from the JSON when possible (labelme records
``imageWidth`` / ``imageHeight``); otherwise opened with OpenCV from the
sibling image file referenced by ``imagePath``.

Usage
-----
    python tasks/object_detection/prepare_dataset.py \
        --in  /path/to/labelme_annotations \
        --out /path/to/yolo_labels

    # convert in-place (writes .txt next to each .json)
    python tasks/object_detection/prepare_dataset.py --in /path/to/data
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running this file directly from the project root.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from tasks.object_detection.packages.dataset_activity import (  # noqa: E402
    CLASSES,
    convert_labelme_json,
)


def _read_dimensions(json_path: Path, data: dict):
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if w and h:
        return int(w), int(h)

    img_rel = data.get("imagePath")
    if not img_rel:
        return None
    img_path = (json_path.parent / img_rel).resolve()
    if not img_path.exists():
        return None

    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return int(w), int(h)


def convert_directory(in_dir: Path, out_dir: Path | None) -> tuple[int, int, int]:
    """Returns (json_seen, txt_written, label_lines)."""
    json_files = sorted(in_dir.rglob("*.json"))
    if not json_files:
        print(f"No .json files found under {in_dir}")
        return 0, 0, 0

    json_seen = 0
    txt_written = 0
    label_lines = 0

    for jp in json_files:
        json_seen += 1
        try:
            with open(jp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {jp}: {e}")
            continue

        dims = _read_dimensions(jp, data)
        if dims is None:
            print(f"  skip {jp}: cannot determine image size")
            continue
        img_w, img_h = dims

        try:
            labels = convert_labelme_json(str(jp), img_w, img_h)
        except Exception as e:
            print(f"  skip {jp}: convert failed: {e}")
            continue

        if out_dir is None:
            txt_path = jp.with_suffix(".txt")
        else:
            rel = jp.relative_to(in_dir).with_suffix(".txt")
            txt_path = out_dir / rel
            txt_path.parent.mkdir(parents=True, exist_ok=True)

        with open(txt_path, "w") as f:
            if labels:
                f.write("\n".join(labels) + "\n")
        txt_written += 1
        label_lines += len(labels)

    return json_seen, txt_written, label_lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", required=True,
                   help="Directory containing labelme .json files (recursive).")
    p.add_argument("--out", dest="out_dir", default=None,
                   help="Output directory. Mirrors the input tree. "
                        "If omitted, .txt files are written next to each .json.")
    args = p.parse_args(argv)

    in_dir = Path(args.in_dir).resolve()
    if not in_dir.is_dir():
        print(f"Error: {in_dir} is not a directory")
        return 2
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    print(f"Classes: {CLASSES}")
    print(f"Input  : {in_dir}")
    print(f"Output : {out_dir or '(in-place, next to each .json)'}")
    print()

    seen, written, lines = convert_directory(in_dir, out_dir)
    print()
    print(f"Done. JSON seen={seen}  TXT written={written}  label lines={lines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
