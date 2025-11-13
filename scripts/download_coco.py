#!/usr/bin/env python3
"""
Utility script to download COCO 2017 dataset, optionally convert annotations to
YOLO-style labels, and evaluate detection results with COCO mAP (pycocotools).

Features:
- Download train2017, val2017 and annotations for COCO 2017.
- Extract archives into a user-specified data directory.
- Optional conversion from COCO annotations to per-image YOLO-format label files.
- Evaluate detection results (COCO-format results JSON) and print mAP

Usage examples:
  # Download datasets into ./data/coco
  python scripts/re-train_yolo_model.py --download --data-dir ./data/coco

  # Convert annotations to YOLO format (creates `images/` and `labels/` under data-dir)
  python scripts/re-train_yolo_model.py --convert-to-yolo --data-dir ./data/coco

  # Compute mAP given COCO ground-truth and result json
  python scripts/re-train_yolo_model.py --eval --ann ./data/coco/annotations/instances_val2017.json --results ./preds/results.json

Notes:
- Converting COCO -> YOLO will create a `labels/` directory with one .txt file per image.
- Evaluation expects detection results in COCO results JSON format (list of dicts with image_id, category_id, bbox, score).
- For conversion/evaluation this script tries to import pycocotools; install with:
	pip install pycocotools

This script is intended as a convenience helper; adapt paths/options to your training/eval pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
	from pycocotools.coco import COCO
	from pycocotools.cocoeval import COCOeval
except Exception:  # pragma: no cover - runtime dependency
	COCO = None
	COCOeval = None

import urllib.request
import zipfile


COCO_URLS = {
	"train_images": "http://images.cocodataset.org/zips/train2017.zip",
	"val_images": "http://images.cocodataset.org/zips/val2017.zip",
	"annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024):
	dest.parent.mkdir(parents=True, exist_ok=True)
	if dest.exists():
		print(f"Already downloaded: {dest}")
		return dest

	print(f"Downloading {url} -> {dest}")
	with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
		total = resp.length if hasattr(resp, "length") else None
		downloaded = 0
		start = time.time()
		while True:
			chunk = resp.read(chunk_size)
			if not chunk:
				break
			out.write(chunk)
			downloaded += len(chunk)
			if total:
				pct = downloaded / total * 100
				print(f"\r{downloaded}/{total} bytes ({pct:.1f}%)", end="")
		if total:
			print()
		print(f"Downloaded {dest} in {time.time()-start:.1f}s")
	return dest


def extract_zip(zip_path: Path, dest_dir: Path):
	print(f"Extracting {zip_path} -> {dest_dir}")
	with zipfile.ZipFile(zip_path, "r") as z:
		z.extractall(dest_dir)
	print("Extraction complete")


def download_coco(data_dir: Path, which: List[str]):
	data_dir = data_dir.resolve()
	downloads = []
	for key in which:
		url = COCO_URLS.get(key)
		if not url:
			raise ValueError(f"Unknown COCO key: {key}")
		dest = data_dir / Path(url).name
		download_file(url, dest)
		downloads.append(dest)
	return downloads


def convert_coco_to_yolo(ann_file: Path, images_dir: Path, out_dir: Path, categories_keep: List[int] = None):
	"""Convert COCO annotations to YOLO per-image .txt labels and a simple data.yml mapping.

	- ann_file: path to instances_*.json
	- images_dir: directory with images (train2017 or val2017)
	- out_dir: will create out_dir/images (symlink or copy), out_dir/labels
	- categories_keep: optional list of category ids to keep; if None, keep all
	"""
	if COCO is None:
		raise RuntimeError("pycocotools is required for conversion. Install: pip install pycocotools")

	ann_file = ann_file.resolve()
	images_dir = images_dir.resolve()
	out_dir = out_dir.resolve()
	out_images = out_dir / "images"
	out_labels = out_dir / "labels"
	out_images.mkdir(parents=True, exist_ok=True)
	out_labels.mkdir(parents=True, exist_ok=True)

	coco = COCO(str(ann_file))

	# Build category id -> class index map
	cats = coco.loadCats(coco.getCatIds())
	cats.sort(key=lambda x: x["id"])  # stable order
	cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

	images = coco.getImgIds()
	print(f"Converting {len(images)} images to YOLO labels in {out_labels}")

	for img_id in images:
		img = coco.loadImgs(img_id)[0]
		file_name = img["file_name"]
		w, h = img["width"], img["height"]
		src_img = images_dir / file_name
		dst_img = out_images / file_name
		if src_img.exists():
			# create a symlink to avoid copying large files if possible
			try:
				if not dst_img.exists():
					os.symlink(src_img, dst_img)
			except Exception:
				# fallback to copy
				if not dst_img.exists():
					shutil.copy2(src_img, dst_img)
		else:
			# Image missing — skip but still write label if annotations exist
			pass

		ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
		anns = coco.loadAnns(ann_ids)
		label_lines = []
		for a in anns:
			cat_id = a["category_id"]
			if categories_keep and cat_id not in categories_keep:
				continue
			bbox = a["bbox"]  # [x,y,width,height] absolute
			# convert to YOLO x_center y_center w h (normalized)
			x, y, bw, bh = bbox
			xc = x + bw / 2.0
			yc = y + bh / 2.0
			xc /= w
			yc /= h
			bw /= w
			bh /= h
			cls_idx = cat_id_to_idx[cat_id]
			label_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

		label_path = out_labels / (Path(file_name).stem + ".txt")
		with open(label_path, "w", encoding="utf-8") as f:
			f.write("\n".join(label_lines))

	# Write a tiny data.yaml for convenience
	data_yaml = {
		"path": str(out_dir),
		"train": "images",
		"val": "images",
		"nc": len(cats),
		"names": [c["name"] for c in cats],
	}
	with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
		json.dump(data_yaml, f, indent=2)

	print(f"Conversion complete. Labels written to {out_labels}, data.yaml at {out_dir / 'data.yaml'}")


def evaluate_coco(ann_file: Path, results_file: Path, iou_type: str = "bbox") -> Dict[str, float]:
	if COCO is None or COCOeval is None:
		raise RuntimeError("pycocotools is required for evaluation. Install: pip install pycocotools")

	ann_file = str(Path(ann_file).resolve())
	results_file = str(Path(results_file).resolve())

	cocoGt = COCO(ann_file)
	cocoDt = cocoGt.loadRes(results_file)
	cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
	cocoEval.params.useCats = 1
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

	# gather a few common metrics
	stats = {
		"AP@[.5:.95]": cocoEval.stats[0],
		"AP@0.5": cocoEval.stats[1],
		"AP@0.75": cocoEval.stats[2],
		"AR@1": cocoEval.stats[6],
		"AR@10": cocoEval.stats[7],
	}
	return stats


def parse_args():
	p = argparse.ArgumentParser(description="Download COCO, convert annotations, and evaluate mAP")
	p.add_argument("--data-dir", type=Path, default=Path("./data/coco"), help="Base data directory")
	p.add_argument("--download", action="store_true",default= False, help="Download COCO archives (train/val/annotations)")
	p.add_argument("--extract-only", action="store_true",default= False, help="Only extract archives if present (no download)")
	p.add_argument("--convert-to-yolo", action="store_true",default= True, help="Convert COCO annotations to YOLO format (labels/images)")
	p.add_argument("--ann", type=Path, help="Path to COCO annotations json (for conversion/eval)")
	p.add_argument("--images-dir", type=Path, help="Path to images directory (for conversion)")
	p.add_argument("--out-dir", type=Path, help="Output directory for converted YOLO dataset (default: <data-dir>/yolo)")
	p.add_argument("--eval", action="store_true",default= False, help="Run COCO evaluation given --ann and --results")
	p.add_argument("--results", type=Path ,default= True, help="COCO results json file to evaluate")
	return p.parse_args()


def main():
	args = parse_args()
	data_dir = args.data_dir
	data_dir.mkdir(parents=True, exist_ok=True)

	# Download and/or extract
	if args.download:
		print("Starting download of COCO archives")
		downloads = download_coco(data_dir, ["train_images", "val_images", "annotations"])
		# extract them after download
		for z in downloads:
			extract_zip(z, data_dir)

	if args.extract_only:
		# try to extract existing zips if present
		print("Extract-only mode: extracting any COCO zip files found in data-dir")
		for name in COCO_URLS.values():
			zname = Path(name).name
			zpath = data_dir / zname
			if zpath.exists():
				extract_zip(zpath, data_dir)

	if args.convert_to_yolo:
		ann = args.ann or (data_dir / "annotations" / "instances_train2017.json")
		images_dir = args.images_dir or (data_dir / "train2017")
		out_dir = args.out_dir or (data_dir / "yolo_train")
		print(f"Converting COCO {ann} with images from {images_dir} -> {out_dir}")
		convert_coco_to_yolo(ann, images_dir, out_dir)

	if args.eval:
		if not args.ann:
			print("--ann is required for --eval", file=sys.stderr)
			sys.exit(2)
		if not args.results:
			print("--results is required for --eval", file=sys.stderr)
			sys.exit(2)
		print(f"Evaluating results {args.results} against {args.ann}")
		stats = evaluate_coco(args.ann, args.results)
		print("Evaluation metrics:")
		for k, v in stats.items():
			print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
	main()
