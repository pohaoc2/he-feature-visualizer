"""
run_cellvit_local.py — local equivalent of notebooks/cellvit_colab_stage2.ipynb.

Takes a zip of H&E patches (256x256 PNGs at the path layout produced by
stages/patchify.py) and a CellViT checkpoint, runs inference on the local GPU,
and writes one JSON per patch to the output directory plus a bundled zip.

Must run inside the dedicated conda env created by scripts/setup_cellvit_local.sh
(pydantic 1.x + CellViT requirements conflict with the main project env).

Usage:
    conda activate cellvit
    python stages/run_cellvit_local.py \
        --zip processed/he.zip \
        --out processed/cellvit \
        --checkpoint ~/checkpoints/CellViT-256.pth \
        --cellvit-repo ~/CellViT

Post-processing (``--postprocess``):
    runner   (default, unchanged, keeps every earlier number reproducible): ImageNet
             normalisation, sigmoid of nuclei_binary_map channel 1 >= 0.5, 3x3
             opening, external contours with area 30-6000 px; the HV maps are not
             used, so touching nuclei stay one object.
    official (opt-in): the CellViT repository's own inference
             (cell_segmentation/inference/cell_detection.py, TIO-IKIM/CellViT):
             normalisation from the checkpoint config
             (transformations.normalize.mean / std), softmax over
             nuclei_binary_map and nuclei_type_map, then
             model.calculate_instance_map(magnification=...) = HoVer-Net style
             HV-map watershed that splits touching nuclei; per-instance type =
             CellViT's majority vote of the type-map argmax inside the instance
             (background -> second most frequent), type_prob = that type's pixel
             fraction. Output JSON has the same keys as the runner path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

CELL_TYPE_NAMES = {
    0: "background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

MIN_NUCLEUS_AREA = 30
MAX_NUCLEUS_AREA = 6000
NUC_THRESHOLD = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, type=Path, help="Input zip of H&E patches (*.png).")
    p.add_argument("--out", required=True, type=Path, help="Output directory for per-patch JSON.")
    p.add_argument("--checkpoint", required=True, type=Path, help="CellViT .pth checkpoint.")
    p.add_argument("--cellvit-repo", required=True, type=Path, help="Path to cloned TIO-IKIM/CellViT repo.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--bundle-zip", type=Path, default=None,
                   help="Optional output zip path. Default: <out>.zip")
    p.add_argument("--patch-glob", default="**/*.png",
                   help="Glob inside extracted zip dir to find patches.")
    p.add_argument("--postprocess", choices=("runner", "official"), default="runner",
                   help="runner (default, legacy) or official CellViT post-processing; see module docstring.")
    p.add_argument("--magnification", type=int, choices=(20, 40), default=40,
                   help="official path only: magnification passed to calculate_instance_map (x40 checkpoints: 40).")
    p.add_argument("--save-instances", type=Path, default=None,
                   help="official path only: also write an .npz with stems and int32 instance maps.")
    return p.parse_args(argv)


def extract_patches(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def strip_prefix(sd: dict, prefix: str = "module.") -> dict:
    if not any(k.startswith(prefix) for k in sd):
        return sd
    return OrderedDict((k[len(prefix):] if k.startswith(prefix) else k, v) for k, v in sd.items())


def build_model(checkpoint_path: Path, cellvit_repo: Path, device: torch.device, return_config: bool = False):
    """Mirror notebook cell 6 (load-model) — CellViT256 / CellViTSAM selection.

    Returns (model, num_nuclei), or (model, num_nuclei, config) with return_config=True."""
    if str(cellvit_repo) not in sys.path:
        sys.path.insert(0, str(cellvit_repo))
    from models.segmentation.cell_segmentation.cellvit import CellViT256, CellViTSAM  # noqa: E402

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    arch = ckpt.get("arch", "CellViT256")
    config = ckpt.get("config", {}) or {}
    data_conf = config.get("data", config.get("dataset", {})) if isinstance(config, dict) else {}
    model_conf = config.get("model", {}) if isinstance(config, dict) else {}
    training_conf = config.get("training", {}) if isinstance(config, dict) else {}

    num_nuclei = int(data_conf.get("num_nuclei_classes", 6))
    num_tissue = int(data_conf.get("num_tissue_classes", 0))
    drop_rate = float(training_conf.get("drop_rate", 0.0))
    attn_drop = float(training_conf.get("attn_drop_rate", 0.0))
    drop_path = float(training_conf.get("drop_path_rate", 0.0))
    pretrained_encoder = model_conf.get("pretrained_encoder", model_conf.get("model_path", ""))

    arch_u = str(arch).upper()
    if "SAM" in arch_u:
        vit_structure = "SAM-H"
        for tag in ("SAM-B", "SAM-L", "SAM-H"):
            if tag in arch_u:
                vit_structure = tag
        model = CellViTSAM(
            model_path=pretrained_encoder,
            num_nuclei_classes=num_nuclei,
            num_tissue_classes=num_tissue,
            vit_structure=vit_structure,
            drop_rate=drop_rate,
        )
    else:
        model = CellViT256(
            model256_path=pretrained_encoder,
            num_nuclei_classes=num_nuclei,
            num_tissue_classes=num_tissue,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path,
        )

    sd = strip_prefix(ckpt["model_state_dict"], "module.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded weights. missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    if return_config:
        return model, num_nuclei, config
    return model, num_nuclei


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def checkpoint_normalisation(config: dict) -> tuple[list[float], list[float]]:
    """mean / std the CellViT repository uses for this checkpoint (cell_detection.py reads
    run_conf["transformations"]["normalize"]; the stored config is flat with dotted keys).
    Falls back to (0.5, 0.5), CellViT's default when the config has no normalize entry."""
    if not isinstance(config, dict):
        return [0.5] * 3, [0.5] * 3
    if "transformations.normalize.mean" in config:
        mean, std = config["transformations.normalize.mean"], config["transformations.normalize.std"]
    else:
        norm = (config.get("transformations", {}) or {}).get("normalize", {}) or {}
        mean, std = norm.get("mean", (0.5, 0.5, 0.5)), norm.get("std", (0.5, 0.5, 0.5))
    return [float(v) for v in mean], [float(v) for v in std]


def official_cell_dict(cell: dict) -> dict:
    """One entry of calculate_instance_map's per-image dict -> the runner's JSON cell format
    (centroid [x, y], contour [[x, y], ...], bbox [[rmin, cmin], [rmax, cmax]])."""
    t = int(cell["type"])
    cx, cy = (float(v) for v in np.asarray(cell["centroid"], dtype=float).ravel()[:2])
    tp = cell.get("type_prob")
    return {
        "centroid": [round(cx, 2), round(cy, 2)],
        "contour": [[int(x), int(y)] for x, y in np.asarray(cell["contour"]).reshape(-1, 2)],
        "bbox": [[int(v) for v in row] for row in np.asarray(cell["bbox"]).reshape(2, 2)],
        "type_cellvit": t,
        "type_name": CELL_TYPE_NAMES.get(t, "unknown"),
        "type_prob": None if tp is None else round(float(tp), 4),
    }


def official_postprocess(model, outputs: dict, magnification: int = 40):
    """CellViT repository post-processing of one batch of raw model outputs.

    Softmax over the binary and type maps (cell_detection.py get_cell_predictions_with_tokens),
    then model.calculate_instance_map. Returns (instance maps (B, H, W) int32, per-image cell lists)."""
    pred = OrderedDict(
        nuclei_binary_map=torch.softmax(outputs["nuclei_binary_map"], dim=1),
        nuclei_type_map=torch.softmax(outputs["nuclei_type_map"], dim=1),
        hv_map=outputs["hv_map"],
    )
    inst, type_preds = model.calculate_instance_map(pred, magnification=magnification)
    inst = np.asarray(inst.cpu().numpy() if hasattr(inst, "cpu") else inst).astype(np.int32)
    cells = [[official_cell_dict(c) for _, c in sorted(d.items(), key=lambda kv: int(kv[0]))] for d in type_preds]
    return inst, cells


class PatchDataset(Dataset):
    def __init__(self, patch_dir: Path, fnames: list[str], transform):
        self.patch_dir = patch_dir
        self.fnames = fnames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int):
        fname = self.fnames[idx]
        img = Image.open(self.patch_dir / fname).convert("RGB")
        return self.transform(img), fname


def detect_cells(pred_map: np.ndarray, num_nuclei: int) -> list[dict]:
    nuc_prob = pred_map[:, :, 0]
    type_pred = pred_map[:, :, 3].astype(np.int32)

    binary = (nuc_prob >= NUC_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_NUCLEUS_AREA or area > MAX_NUCLEUS_AREA:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        mask = np.zeros(nuc_prob.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        type_vals = type_pred[mask == 1]
        cell_type = int(np.bincount(type_vals.clip(0, num_nuclei - 1)).argmax())

        x, y, w, h = cv2.boundingRect(contour)
        prob_at_centroid = float(nuc_prob[int(cy), int(cx)])

        cells.append({
            "centroid": [round(cx, 2), round(cy, 2)],
            "contour": contour.reshape(-1, 2).tolist(),
            "bbox": [[y, x], [y + h, x + w]],
            "type_cellvit": cell_type,
            "type_name": CELL_TYPE_NAMES.get(cell_type, "unknown"),
            "type_prob": round(prob_at_centroid, 4),
        })
    return cells


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    official = args.postprocess == "official"
    if args.save_instances is not None and not official:
        raise SystemExit("--save-instances needs --postprocess official")
    args.out.mkdir(parents=True, exist_ok=True)
    bundle_zip = args.bundle_zip or args.out.with_suffix(".zip")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected — inference will be very slow.")

    with tempfile.TemporaryDirectory(prefix="cellvit_he_") as tmp:
        patch_root = extract_patches(args.zip, Path(tmp))
        patches = sorted(p.relative_to(patch_root).as_posix()
                         for p in patch_root.glob(args.patch_glob))
        print(f"Extracted {len(patches)} patches to {patch_root}")
        assert patches, "No .png patches found in zip."

        if args.skip_existing:
            done = {p.stem for p in args.out.glob("*.json")}
            todo = [f for f in patches if Path(f).stem not in done]
            print(f"Skipping {len(done)} already-processed patches.")
        else:
            todo = patches
        print(f"Patches to process: {len(todo)}")

        model, num_nuclei, config = build_model(args.checkpoint, args.cellvit_repo, device, return_config=True)

        if official:
            mean, std = checkpoint_normalisation(config)
            print(f"Post-processing: official (normalise mean={mean} std={std}, softmax, "
                  f"HV watershed, magnification={args.magnification})")
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
            print("Post-processing: runner (legacy)")
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        ds = PatchDataset(patch_root, todo, tfm)
        loader = DataLoader(ds, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"),
                            shuffle=False)

        total_cells = 0
        inst_stems, inst_maps = [], []
        with torch.inference_mode():
            for imgs, fnames in tqdm(loader, desc="Patches"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                if official:
                    inst, cells_b = official_postprocess(model, outputs, args.magnification)
                    for b, fname in enumerate(fnames):
                        patch_id = Path(fname).stem
                        with open(args.out / f"{patch_id}.json", "w") as f:
                            json.dump({"patch": patch_id, "cells": cells_b[b]}, f)
                        total_cells += len(cells_b[b])
                    if args.save_instances is not None:
                        inst_stems += [Path(f).stem for f in fnames]
                        inst_maps.append(inst)
                    continue
                nuc_prob = torch.sigmoid(outputs["nuclei_binary_map"])[:, 1, :, :]
                hv_x = outputs["hv_map"][:, 0, :, :]
                hv_y = outputs["hv_map"][:, 1, :, :]
                type_pred = torch.argmax(
                    torch.softmax(outputs["nuclei_type_map"], dim=1), dim=1
                ).float()
                pred_maps = torch.stack([nuc_prob, hv_x, hv_y, type_pred], dim=-1)
                pred_maps = pred_maps.cpu().numpy().astype(np.float32)

                for b, fname in enumerate(fnames):
                    patch_id = Path(fname).stem
                    cells = detect_cells(pred_maps[b], num_nuclei)
                    with open(args.out / f"{patch_id}.json", "w") as f:
                        json.dump({"patch": patch_id, "cells": cells}, f)
                    total_cells += len(cells)

        if args.save_instances is not None:
            args.save_instances.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_instances, stems=np.array(inst_stems),
                                inst=np.concatenate(inst_maps) if inst_maps else np.zeros((0, 256, 256), np.int32))
        n = max(len(todo), 1)
        print(f"\nDone. {len(todo)} patches, {total_cells:,} cells total. "
              f"Avg cells/patch: {total_cells / n:.1f}")

    # Bundle outputs
    json_files = sorted(args.out.glob("*.json"))
    if json_files:
        if bundle_zip.exists():
            bundle_zip.unlink()
        with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for jf in tqdm(json_files, desc="Zipping"):
                zf.write(jf, arcname=f"cellvit/{jf.name}")
        size_mb = bundle_zip.stat().st_size / (1024 ** 2)
        print(f"Bundle: {bundle_zip} ({size_mb:.2f} MB, {len(json_files)} files)")


if __name__ == "__main__":
    main()
