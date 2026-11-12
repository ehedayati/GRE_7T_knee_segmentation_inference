#!/usr/bin/env python3
"""Segment one original multi-echo GRE NIfTI with a preserved joint cartilage/meniscus model.

Loads inference settings and verifies the checkpoint hash from manifest.json
beside the model. No training CSV or project internal_libs are needed.
Outputs raw labels: 0 background, 1 femoral cartilage, 2 tibial cartilage,
3 and 4 menisci (medial/lateral mapping not yet confirmed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
# Shared release-local architecture; also works when called from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_unet import UNet3D


def load_settings(model_path: Path):
    manifest_path = model_path.with_name("manifest.json")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {manifest_path}")
    matches = [entry for entry in manifest["models"]
               if entry["checkpoint"] == model_path.name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest entry for {model_path.name}")
    entry = matches[0]
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {model_path}")
    cfg = SimpleNamespace(**entry["inference"])
    if cfg.Network not in ("UNet", "UNet2") or cfg.out_channel != 5:
        raise ValueError("This runner supports the preserved five-class UNet/UNet2 models")
    if cfg.num_echos != len(cfg.specific_echos) or cfg.num_echos < 1:
        raise ValueError("Manifest echo count does not match its echo indices")
    if cfg.crop_size <= 0 or cfg.window_size <= 0:
        raise ValueError("Manifest crop and window sizes must be positive")
    print(f"Verified SHA-256: {model_path.name}")
    return cfg


def load_model(cfg, model_path: Path, device: torch.device):
    # Match internal_libs/models/models.py exactly for the two archived families.
    channels, strides = {
        "UNet": ((16, 32, 64, 128, 256), (2, 2, 2, 2)),
        "UNet2": ((16, 32, 64, 128, 128), (2, 2, 2, 1)),
    }[cfg.Network]
    model = UNet3D(
        in_channels=cfg.num_echos,
        out_channels=cfg.out_channel,
        channels=channels,
        strides=strides,
        num_res_units=cfg.Unet_res_no,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    print(f"Loaded {cfg.Network} state dict: {model_path}")
    return model


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("The model file is not a PyTorch state-dict dictionary")
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def prepare_image(raw: np.ndarray, echoes: list[int], cfg) -> np.ndarray:
    if raw.ndim == 3:
        raw = raw[..., None]
    if raw.ndim != 4:
        raise ValueError(f"Expected a 3-D/4-D NIfTI, got shape {raw.shape}")
    if not echoes or min(echoes) < 0 or max(echoes) >= raw.shape[-1]:
        raise ValueError(f"Echoes {echoes} invalid for {raw.shape[-1]} input echoes")

    image = np.asarray(raw[..., echoes], dtype=np.float32).copy()
    if bool(getattr(cfg, "normalize", False)):
        for channel in range(image.shape[-1]):
            values = image[..., channel]
            if bool(getattr(cfg, "z_score", False)):
                scale = float(values.std())
            else:
                scale = float(values.max() - values.min())
            image[..., channel] = (values - float(values.mean())) / max(scale, 1e-8)
    return image


@torch.inference_mode()
def segment(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    crop_size: int,
    window_size: int,
    classes: int,
) -> np.ndarray:
    """Apply the training-time interleaved quadrants, center crop, and slice blocks."""
    if crop_size <= 0 or window_size <= 0:
        raise ValueError("Crop and window sizes must be positive")
    output = np.zeros(image.shape[:3], np.uint16 if classes > 255 else np.uint8)

    for quadrant in range(4):
        row, column = divmod(quadrant, 2)
        view = image[row::2, column::2]
        if min(view.shape[:2]) < crop_size:
            raise ValueError(f"Quadrant {view.shape[:2]} is smaller than crop {crop_size}")
        x0 = (view.shape[0] - crop_size) // 2
        y0 = (view.shape[1] - crop_size) // 2
        cropped = view[x0:x0 + crop_size, y0:y0 + crop_size]
        quadrant_output = np.zeros(view.shape[:3], output.dtype)

        for z0 in range(0, cropped.shape[2], window_size):
            valid = min(window_size, cropped.shape[2] - z0)
            block = np.zeros(
                (crop_size, crop_size, window_size, image.shape[-1]), np.float32
            )
            block[:, :, :valid] = cropped[:, :, z0:z0 + valid]
            tensor = (
                torch.from_numpy(block).permute(3, 2, 0, 1).unsqueeze(0).to(device)
            )
            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if isinstance(logits, dict):
                logits = logits.get("out", next(iter(logits.values())))
            labels = torch.argmax(logits, dim=1)[0, :valid].cpu().numpy()
            quadrant_output[
                x0:x0 + crop_size, y0:y0 + crop_size, z0:z0 + valid
            ] = np.moveaxis(labels, 0, 2)

        output[row::2, column::2] = quadrant_output
    return output


def default_output(path: Path) -> Path:
    base = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
    return path.with_name(base + "_seg.nii.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Default: INPUT_seg.nii.gz")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_path = (args.output or default_output(input_path)).expanduser().resolve()
    for path in (model_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path in (input_path, model_path, model_path.with_name("manifest.json")):
        raise ValueError("Output must not replace the input, checkpoint, or manifest")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; choose a new path or use --overwrite: {output_path}")
    if not output_path.name.lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Output must end in .nii or .nii.gz")

    cfg = load_settings(model_path)
    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    nii = nib.load(str(input_path))
    raw = np.asanyarray(nii.dataobj)
    image = prepare_image(raw, list(cfg.specific_echos), cfg)
    if min(image.shape[:2]) < 2 * cfg.crop_size or image.shape[2] < 1:
        raise ValueError(f"Input needs X/Y >= {2 * cfg.crop_size} and nonzero depth; got {image.shape}")
    print(f"Input {raw.shape}; echoes {cfg.specific_echos}; device {device}")
    model = load_model(cfg, model_path, device)
    labels = segment(image, model, device, cfg.crop_size, cfg.window_size, cfg.out_channel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = nii.header.copy()
    header.set_data_dtype(labels.dtype)
    nib.save(nib.Nifti1Image(labels, nii.affine, header), str(output_path))
    print(f"Saved segmentation: {output_path}")


if __name__ == "__main__":
    main()
