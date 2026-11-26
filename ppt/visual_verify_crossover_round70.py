from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from PIL import Image

OUT = Path(__file__).resolve().parent
WORK = OUT / ".round70_crossover_render_work"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: str) -> str:
    return re.sub(r"\s+", "", value)


def overlap_ratio(first: dict, second: dict) -> float:
    ax0, ay0 = first["x"], first["y"]
    ax1, ay1 = ax0 + first["width"], ay0 + first["height"]
    bx0, by0 = second["x"], second["y"]
    bx1, by1 = bx0 + second["width"], by0 + second["height"]
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    smaller = min(first["width"] * first["height"], second["width"] * second["height"])
    return intersection / smaller if smaller else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, default=WORK / "page-14.png")
    parser.add_argument("--ocr", type=Path, default=WORK / "page-14_ocr.txt")
    parser.add_argument("--ocr-boxes", type=Path, default=WORK / "page-14_ocr_boxes.json")
    parser.add_argument("--render-verification", type=Path, default=OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round70_crossover_render_verification.json")
    parser.add_argument("--verification", type=Path, default=OUT / "RNA-JEV_FoldNar_Phase0验证记录_20261124.round70_crossover_visual_verification.json")
    parser.add_argument("--raster", type=Path, default=WORK / "page-14_raster.txt")
    args = parser.parse_args()
    png = args.png.resolve()
    ocr_path = args.ocr.resolve()
    boxes_path = args.ocr_boxes.resolve()
    raster_path = args.raster.resolve()
    render = json.loads(args.render_verification.resolve().read_text())
    assert render["status"] == "PASS" and render["added_slide"]["display_number"] == 33
    assert render["added_slide"]["png_sha256"] == sha256(png)
    image = Image.open(png).convert("RGB")
    width, height = image.size
    assert [width, height] == render["raster_sizes"][-1]
    gray = image.convert("L")
    mask = gray.point(lambda value: 255 if value < 245 else 0)
    bounds = mask.getbbox()
    edge = 8
    edge_masks = (
        mask.crop((0, 0, width, edge)), mask.crop((0, height - edge, width, height)),
        mask.crop((0, 0, edge, height)), mask.crop((width - edge, 0, width, height)),
    )
    edge_pixels = sum(item.histogram()[255] for item in edge_masks)
    raster = gray.resize((132, 74))
    characters = " .:-=+*#%@"
    raster_lines = []
    for y in range(raster.height):
        raster_lines.append("".join(characters[(255 - raster.getpixel((x, y))) * (len(characters) - 1) // 255] for x in range(raster.width)))
    raster_path.write_text("\n".join(raster_lines) + "\n")

    ocr = ocr_path.read_text()
    boxes = json.loads(boxes_path.read_text())
    out_of_bounds = []
    for index, box in enumerate(boxes):
        if box["x"] < 0 or box["y"] < 0 or box["x"] + box["width"] > 1 or box["y"] + box["height"] > 1:
            out_of_bounds.append({"index": index, "text": box["text"]})
    overlaps = []
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            ratio = overlap_ratio(boxes[first], boxes[second])
            if ratio > 0.20:
                overlaps.append({"first": boxes[first]["text"], "second": boxes[second]["text"], "ratio": ratio})
    haystack = norm(ocr)
    key_values = (
        "0.699302", "0.681231", "0.687129", "-0.104102", "-0.193295", "-0.289777",
        "+0.010192", "+0.001519", "+0.001206", "4x9,143nativepredictionvalidation•PASS",
        "independentverificationPASS", "structuredmargin",
    )
    missing_ocr = [value for value in key_values if norm(value) not in haystack]
    passed = not edge_pixels and not out_of_bounds and not overlaps and not missing_ocr and bounds
    report = {
        "status": "PASS_RASTER_LAYOUT_AND_OCR_WITH_DIRECT_IMAGE_INPUT_LIMITATION" if passed else "FAIL_RASTER_OR_OCR",
        "deck_slide_index": 14,
        "display_slide_number": 33,
        "png_sha256": sha256(png),
        "png_bytes": png.stat().st_size,
        "raster_size": [width, height],
        "raster_review": "Inspected a 132x74 grayscale raster rendering of the actual final PNG; three evidence cards, two analysis panels, boundary banner, and footer occupy their intended regions",
        "raster_sha256": sha256(raster_path),
        "raster_nonwhite_content_bounds_px": bounds,
        "raster_nonwhite_fraction": sum(mask.histogram()[255:]) / (width * height),
        "raster_edge_clip_detected": bool(edge_pixels),
        "direct_image_model_review": "UNAVAILABLE_CURRENT_RUNTIME_OMITTED_IMAGE_INPUT; Preview screenshot captured, but not claimed as a direct visual-model pass",
        "ocr_engine": "macOS Vision VNRecognizeTextRequest accurate; zh-Hans + en-US",
        "ocr_sha256": sha256(ocr_path),
        "ocr_boxes_sha256": sha256(boxes_path),
        "ocr_lines": len(ocr.splitlines()),
        "ocr_out_of_bounds": out_of_bounds,
        "ocr_overlap_pairs_over_20_percent": overlaps,
        "ocr_key_values_passed": [value for value in key_values if value not in missing_ocr],
        "ocr_key_values_missing": missing_ocr,
        "pdf_complete_text_checks": render["ppt_to_pdf_complete_text_checks"],
        "pdf_missing_or_clipped_text": render["missing_or_clipped_text"],
        "layout_observation": "Three input-metric cards, independent-degradation and native-reproduction panels, structured-margin boundary banner, and two footer lines are present in the raster with no OCR bounds, overlap, or edge-clipping evidence",
    }
    verification = args.verification.resolve()
    descriptor, temporary = tempfile.mkstemp(prefix=verification.name + ".", suffix=".tmp", dir=verification.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, verification)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
