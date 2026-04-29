#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--left",
        type=str,
        default="./alignment_js_vs_conf_fmnist_pretty.png",
    )
    p.add_argument(
        "--right",
        type=str,
        default="./alignment_js_vs_conf_cifar10_pretty.png",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./js_divergence_comparison.pdf",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape={img.shape}")
    h, w, c = img.shape
    if h == target_h:
        return img
    if h > target_h:
        return img[:target_h, :, :]
    pad_total = target_h - h
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    if np.issubdtype(img.dtype, np.floating):
        bg = 1.0
    else:
        bg = np.array(255, dtype=img.dtype)
    padded = np.full((target_h, w, c), bg, dtype=img.dtype)
    padded[pad_top : pad_top + h, :, :] = img
    return padded


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape={img.shape}")
    if img.shape[2] == 4:
        rgb = img[:, :, :3]
        a = img[:, :, 3:4]
        if np.issubdtype(img.dtype, np.floating):
            bg = 1.0
        else:
            bg = np.array(255, dtype=img.dtype)
        return rgb * a + bg * (1 - a)
    return img


def main():
    args = _parse_args()
    left = _ensure_rgb(plt.imread(args.left))
    right = _ensure_rgb(plt.imread(args.right))
    target_h = int(max(left.shape[0], right.shape[0]))
    left = _pad_to_height(left, target_h)
    right = _pad_to_height(right, target_h)

    if np.issubdtype(left.dtype, np.floating):
        bg = 1.0
    else:
        bg = np.array(255, dtype=left.dtype)
    canvas = np.full((target_h, left.shape[1] + right.shape[1], 3), bg, dtype=left.dtype)
    canvas[:, : left.shape[1], :] = left[:, :, :3]
    canvas[:, left.shape[1] :, :] = right[:, :, :3]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
        }
    )
    fig_w = canvas.shape[1] / float(args.dpi)
    fig_h = canvas.shape[0] / float(args.dpi)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=args.dpi)
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(args.output, format="pdf", dpi=args.dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
