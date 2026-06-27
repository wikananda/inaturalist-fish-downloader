"""Benchmark YOLO and SAM 3 fish cropping on existing raw images."""

import argparse
from pathlib import Path
from typing import Any

from ..common.utils import parse_csv_int_set, parse_csv_set, safe_print, slugify
from ..download.crop_benchmark import (
    CropResult,
    box_iou,
    load_benchmark_samples,
    make_contact_sheet,
    run_sam3_cropper,
    run_yolo_cropper,
    write_metrics_csv,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    """Parse cropper benchmark CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO and SAM 3 fish croppers on existing raw images."
    )
    parser.add_argument("--manifest", default="manifests/candidates.jsonl")
    parser.add_argument("--raw-dir", default="downloads_raw")
    parser.add_argument("--output-dir", default="benchmarks/croppers")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--max-per-species", type=int, default=10)
    parser.add_argument(
        "--backends",
        choices=["both", "yolo", "sam3"],
        default="both",
        help="Which cropper backends to benchmark.",
    )
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--contact-sheets", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--yolo-weights", default="models/fish-yolo.pt")
    parser.add_argument("--yolo-confidence", type=float, default=0.5)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-class-ids", default=None)
    parser.add_argument("--yolo-class-names", default=None)
    parser.add_argument("--min-fish-area-ratio", type=float, default=0.02)

    parser.add_argument("--sam-prompt", default="fish")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate cropper benchmark arguments."""
    if args.max_images <= 0:
        raise SystemExit("--max-images must be greater than 0")
    if args.max_per_species <= 0:
        raise SystemExit("--max-per-species must be greater than 0")
    if args.crop_padding < 0:
        raise SystemExit("--crop-padding must be greater than or equal to 0")
    if args.yolo_confidence < 0 or args.yolo_confidence > 1:
        raise SystemExit("--yolo-confidence must be between 0 and 1")
    if args.yolo_imgsz <= 0:
        raise SystemExit("--yolo-imgsz must be greater than 0")
    if args.min_fish_area_ratio < 0 or args.min_fish_area_ratio > 1:
        raise SystemExit("--min-fish-area-ratio must be between 0 and 1")

    try:
        args.yolo_class_id_set = parse_csv_int_set(args.yolo_class_ids)
    except ValueError as exc:
        raise SystemExit("--yolo-class-ids must be comma-separated integers") from exc
    args.yolo_class_name_set = parse_csv_set(args.yolo_class_names)


def main() -> None:
    """Run the cropper benchmark."""
    args = parse_args()
    validate_args(args)

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_benchmark_samples(
        manifest_path=manifest_path,
        raw_dir=raw_dir,
        max_images=args.max_images,
        max_per_species=args.max_per_species,
    )
    if not samples:
        raise SystemExit(
            f"No benchmark images found from {manifest_path} or {raw_dir}"
        )

    backends = ["yolo", "sam3"] if args.backends == "both" else [args.backends]
    rows: list[dict[str, Any]] = []
    safe_print(f"Benchmarking {len(samples)} images with {', '.join(backends)}")

    for index, sample in enumerate(samples, start=1):
        safe_print(f"[{index}/{len(samples)}] {sample.species}: {sample.raw_path.name}")
        results: dict[str, CropResult] = {}
        for backend in backends:
            crop_path = (
                output_dir
                / "crops"
                / backend
                / slugify(sample.species)
                / sample.raw_path.name
            )
            if backend == "yolo":
                result = run_yolo_cropper(
                    image_path=sample.raw_path,
                    output_path=crop_path,
                    weights=args.yolo_weights,
                    device=args.device,
                    confidence=args.yolo_confidence,
                    imgsz=args.yolo_imgsz,
                    class_ids=args.yolo_class_id_set,
                    class_names=args.yolo_class_name_set,
                    min_area_ratio=args.min_fish_area_ratio,
                    crop_padding=args.crop_padding,
                )
            else:
                result = run_sam3_cropper(
                    image_path=sample.raw_path,
                    output_path=crop_path,
                    prompt=args.sam_prompt,
                    device=args.device,
                    crop_padding=args.crop_padding,
                )
            results[backend] = result

        ious = _backend_ious(results)
        for backend, result in results.items():
            row = _row_for_result(sample, result)
            row["box_iou_with_other"] = ious.get(backend)
            rows.append(row)

        if args.contact_sheets:
            make_contact_sheet(
                raw_path=sample.raw_path,
                crop_paths={backend: result.crop_path for backend, result in results.items()},
                output_path=output_dir
                / "contact_sheets"
                / slugify(sample.species)
                / f"{sample.raw_path.stem}.jpg",
            )

    write_metrics_csv(output_dir / "metrics.csv", rows)
    write_summary_json(output_dir / "summary.json", rows)
    safe_print(f"Wrote benchmark results to {output_dir}")


def _backend_ious(results: dict[str, CropResult]) -> dict[str, float | None]:
    if "yolo" not in results or "sam3" not in results:
        return {}
    iou = box_iou(results["yolo"].crop_box_xyxy, results["sam3"].crop_box_xyxy)
    return {"yolo": iou, "sam3": iou}


def _row_for_result(sample, result: CropResult) -> dict[str, Any]:
    crop_box = ""
    if result.crop_box_xyxy is not None:
        crop_box = ",".join(str(value) for value in result.crop_box_xyxy)
    return {
        "image_id": sample.image_id,
        "species": sample.species,
        "raw_path": str(sample.raw_path),
        "backend": result.backend,
        "success": result.success,
        "reject_reason": result.reject_reason,
        "inference_seconds": round(result.inference_seconds, 6),
        "detection_count": result.detection_count,
        "score": _round_optional(result.score),
        "crop_area_ratio": _round_optional(result.crop_area_ratio),
        "mask_area_ratio": _round_optional(result.mask_area_ratio),
        "crop_box_xyxy": crop_box,
        "crop_path": str(result.crop_path) if result.crop_path else "",
    }


def _round_optional(value: float | None) -> float | str:
    if value is None:
        return ""
    return round(float(value), 6)


if __name__ == "__main__":
    main()
