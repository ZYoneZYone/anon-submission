#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from nets import resnet18, resnet34
from utils import get_dataset, setup_seed


TITLE_FONTSIZE = 22
TITLE_MAIN_FONTSIZE = 16
TITLE_DATASET_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 16


def _set_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "axes.linewidth": 1.2,
            "grid.alpha": 0.3,
            "lines.linewidth": 2.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["mnist", "fmnist", "cifar10", "svhn"])
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--target_arch", type=str, default="resnet34", choices=["resnet34", "resnet18"])
    p.add_argument("--target_weight_path", type=str, required=True)

    p.add_argument("--student_arch", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    p.add_argument(
        "--student_no_posttrain_weight_path",
        type=str,
        required=True,
    )
    p.add_argument(
        "--student_naive_posttrain_weight_path",
        type=str,
        required=True,
    )
    p.add_argument(
        "--student_addt_weight_path",
        type=str,
        required=True,
    )

    p.add_argument("--bins", type=int, default=18)
    p.add_argument("--figure_dpi", type=int, default=300)
    p.add_argument("--output_path", type=str, default="./js_divergence_vs_conf.png")
    return p.parse_args()


def _build_model(arch: str, dataset: str) -> torch.nn.Module:
    if arch == "resnet18":
        return resnet18(dataset)
    if arch == "resnet34":
        return resnet34(dataset)
    raise ValueError(f"Unsupported arch: {arch}")


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if "model_state_dict" in state_dict and isinstance(state_dict["model_state_dict"], dict):
        state_dict = state_dict["model_state_dict"]
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        candidate = state_dict["model"]
        if all(torch.is_tensor(v) for v in candidate.values()):
            state_dict = candidate

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def _load_weights_into(model: torch.nn.Module, weight_path: str) -> torch.nn.Module:
    if not os.path.exists(weight_path):
        raise FileNotFoundError(weight_path)
    obj = torch.load(weight_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported weight format: {type(obj)} at {weight_path}")
    state_dict = _clean_state_dict(obj)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    return model


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    labels = []
    seen = 0
    for x, y in loader:
        if max_samples and seen >= max_samples:
            break
        take = x.shape[0]
        if max_samples:
            take = min(take, max_samples - seen)
        x = x[:take].to(device, non_blocking=True)
        y = (y[:take] % 10).to(device, non_blocking=True)
        out = model(x).detach().cpu().numpy()
        logits.append(out)
        labels.append(y.detach().cpu().numpy())
        seen += take
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def _softmax(x: np.ndarray, t: float) -> np.ndarray:
    x = x / max(float(t), 1e-8)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=1)
    return 0.5 * (kl_pm + kl_qm)


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    xs = np.sort(x)
    ys = np.linspace(0.0, 1.0, xs.shape[0], endpoint=True)
    return xs, ys


def _binned_stats(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    edges = np.linspace(x.min(), x.max(), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full((bins,), np.nan, dtype=np.float64)
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i + 1]) if i < bins - 1 else (x >= edges[i]) & (x <= edges[i + 1])
        if m.any():
            out[i] = float(y[m].mean())
    return centers, out


def _style_ax(ax):
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    ax.tick_params(axis="both", which="major", length=4, width=1.0, labelsize=TICK_LABEL_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def main():
    args = _parse_args()
    setup_seed(args.seed)
    _set_plot_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train_loader, test_loader = get_dataset(args.dataset)
    test_dataset = test_loader.dataset
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    target = _load_weights_into(_build_model(args.target_arch, args.dataset), args.target_weight_path).to(device)
    no_post = _load_weights_into(_build_model(args.student_arch, args.dataset), args.student_no_posttrain_weight_path).to(device)
    naive_post = _load_weights_into(_build_model(args.student_arch, args.dataset), args.student_naive_posttrain_weight_path).to(device)
    addt = _load_weights_into(_build_model(args.student_arch, args.dataset), args.student_addt_weight_path).to(device)

    t_logits, y_true = _collect_logits(target, loader, device, args.max_samples)
    s0_logits, _ = _collect_logits(no_post, loader, device, args.max_samples)
    s1_logits, _ = _collect_logits(naive_post, loader, device, args.max_samples)
    s2_logits, _ = _collect_logits(addt, loader, device, args.max_samples)

    t_prob = _softmax(t_logits, args.temperature)
    s0_prob = _softmax(s0_logits, args.temperature)
    s1_prob = _softmax(s1_logits, args.temperature)
    s2_prob = _softmax(s2_logits, args.temperature)

    t_pred = t_prob.argmax(1).astype(np.int64)
    s0_pred = s0_prob.argmax(1).astype(np.int64)
    s1_pred = s1_prob.argmax(1).astype(np.int64)
    s2_pred = s2_prob.argmax(1).astype(np.int64)

    target_acc = float((t_pred == (y_true.astype(np.int64) % 10)).mean() * 100.0)
    t_conf = t_prob.max(axis=1)

    s0_js = _js_divergence(s0_prob, t_prob)
    s1_js = _js_divergence(s1_prob, t_prob)
    s2_js = _js_divergence(s2_prob, t_prob)

    s0_dis = (s0_pred != t_pred).astype(np.float64)
    s1_dis = (s1_pred != t_pred).astype(np.float64)
    s2_dis = (s2_pred != t_pred).astype(np.float64)

    colors = {
        "no_post": "#d62728",
        "naive": "#ff7f0e",
        "addt": "#1f77b4",
    }

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.8), dpi=args.figure_dpi)
    centers, s0_js_b = _binned_stats(t_conf, s0_js, args.bins)
    _c, s1_js_b = _binned_stats(t_conf, s1_js, args.bins)
    _c, s2_js_b = _binned_stats(t_conf, s2_js, args.bins)
    step = max(1, int(len(centers) // 12))
    ax.plot(
        centers,
        s0_js_b,
        color=colors["no_post"],
        label="No post-train",
        marker="o",
        markersize=5.0,
        markevery=step,
        alpha=0.95,
        zorder=3,
    )
    ax.plot(
        centers,
        s1_js_b,
        color=colors["naive"],
        label="Naive post-train",
        marker="o",
        markersize=5.0,
        markevery=step,
        alpha=0.95,
        zorder=3,
    )
    ax.plot(
        centers,
        s2_js_b,
        color=colors["addt"],
        label="ADDT",
        marker="o",
        markersize=5.0,
        markevery=step,
        alpha=0.98,
        zorder=4,
    )
    ax.set_title(
        "JS Divergence vs Target Confidence",
        pad=18,
        fontsize=TITLE_MAIN_FONTSIZE,
        fontweight="bold",
        y=1.06,
    )
    ax.text(
        0.5,
        1.01,
        f"({args.dataset.upper()})",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=TITLE_DATASET_FONTSIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Target confidence (max prob)", fontweight="bold")
    ax.set_ylabel("Mean JS", fontweight="bold")
    ax.set_xlim(float(np.nanmin(centers)), 1.0)
    y_all = np.concatenate([s0_js_b, s1_js_b, s2_js_b], axis=0)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size > 0:
        y_min = float(y_all.min())
        y_max = float(y_all.max())
        pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.05
        ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        frameon=True,
        edgecolor="gray",
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        handlelength=2.2,
        borderpad=0.5,
        labelspacing=0.4,
        handletextpad=0.6,
        borderaxespad=0.6,
    )
    _style_ax(ax)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.output_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
