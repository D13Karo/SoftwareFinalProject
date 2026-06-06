import json
from typing import List

CLASSES = ['duckie', 'truck', 'sign']
IMAGE_SIZE = 416


def convert_labelme_json(json_path: str, img_w: int, img_h: int) -> List[str]:
    with open(json_path) as f:
        data = json.load(f)

    labels = []

    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in CLASSES:
            continue

        cls_id = CLASSES.index(label)

        (x0, y0), (x1, y1) = shape["points"]
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        xmin = xmin * IMAGE_SIZE / img_w
        xmax = xmax * IMAGE_SIZE / img_w
        ymin = ymin * IMAGE_SIZE / img_h
        ymax = ymax * IMAGE_SIZE / img_h

        cx = (xmin + xmax) / 2 / IMAGE_SIZE
        cy = (ymin + ymax) / 2 / IMAGE_SIZE
        w  = (xmax - xmin) / IMAGE_SIZE
        h  = (ymax - ymin) / IMAGE_SIZE

        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return labels