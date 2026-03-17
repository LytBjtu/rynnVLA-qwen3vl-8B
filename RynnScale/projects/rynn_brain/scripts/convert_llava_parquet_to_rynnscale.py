import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


def _iter_parquet_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No .parquet files found under: {input_path}")
        return files

    if any(ch in input_path for ch in "*?[]"):
        files = sorted(glob.glob(input_path))
        if not files:
            raise FileNotFoundError(f"No files matched glob: {input_path}")
        return files

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.endswith(".parquet"):
        raise ValueError("--input must be a parquet file, a directory, or a glob like '/path/*.parquet'")
    return [input_path]


def _safe_rel_path(path: str) -> str:
    # Keep directory structure but prevent absolute paths / path traversal.
    path = path.replace("\\", "/")
    path = path.lstrip("/")
    parts = [p for p in path.split("/") if p not in ("", ".", "..")]
    if not parts:
        return "unknown.jpg"
    return "/".join(parts)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_jsonl(rows: Iterable[Dict[str, Any]], output_path: str) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(output_path)) or ".")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LLaVA-style caption parquet shards (id/image/caption) to RynnScale VLMDataset jsonl with exported images."
    )
    parser.add_argument(
        "--input",
        default="LLaVA",
        help="Input parquet directory, parquet file, or glob (default: LLaVA)",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Output jsonl path (e.g. /tmp/rynn_llava/train.jsonl)",
    )
    parser.add_argument(
        "--output_image_dir",
        required=True,
        help="Directory to export images (e.g. /tmp/rynn_llava/images)",
    )
    parser.add_argument(
        "--prompt",
        default="Describe the image.",
        help="User prompt appended after the image",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: convert only first N samples (0 = all)",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality when exporting .jpg files",
    )
    args = parser.parse_args()

    parquet_files = _iter_parquet_files(args.input)
    ds = load_dataset("parquet", data_files=parquet_files)["train"]

    required_cols = {"id", "image", "caption"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise KeyError(f"Input dataset missing columns: {sorted(missing)}; got: {ds.column_names}")

    _ensure_dir(args.output_image_dir)

    rows: List[Dict[str, Any]] = []
    n = len(ds) if args.limit <= 0 else min(len(ds), args.limit)

    for i in range(n):
        ex = ds[i]
        image_id: str = ex["id"]
        caption: str = ex["caption"]
        image = ex["image"]  # usually a PIL.Image

        rel_path = _safe_rel_path(image_id)
        # Preserve extension if present, else default to .jpg
        root, ext = os.path.splitext(rel_path)
        if ext.lower() not in {".jpg", ".jpeg", ".png"}:
            rel_path = rel_path + ".jpg"
            ext = ".jpg"

        out_path = os.path.join(args.output_image_dir, rel_path)
        _ensure_dir(os.path.dirname(out_path))

        # Export image to disk so training can load it by path.
        if ext.lower() in {".jpg", ".jpeg"}:
            image.convert("RGB").save(out_path, format="JPEG", quality=args.jpeg_quality)
        else:
            image.save(out_path)

        width, height = image.size

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": out_path, "height": height, "width": width},
                    {"type": "text", "text": args.prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption},
                ],
            },
        ]

        rows.append(
            {
                "conversation": conversation,
                # Needed when using decoder_load_balancing/dynamic_batching. A safe constant is enough for smoke tests.
                "text_sequence_length": 128,
            }
        )

    _write_jsonl(rows, args.output_jsonl)
    print(f"Converted {len(rows)} samples")
    print(f"Wrote jsonl: {args.output_jsonl}")
    print(f"Exported images under: {args.output_image_dir}")


if __name__ == "__main__":
    main()
