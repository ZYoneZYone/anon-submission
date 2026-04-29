#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from nets import resnet18, resnet34
from utils import get_dataset, setup_seed


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="svhn", choices=["svhn"])
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_batches", type=int, default=0)

    p.add_argument("--target_arch", type=str, default="resnet34", choices=["resnet34", "resnet18"])
    p.add_argument("--target_weight_path", type=str, default="./target_model_weight/resnet34_svhn.pth")

    p.add_argument("--student_arch", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    p.add_argument(
        "--student_no_posttrain_weight_path",
        type=str,
        default="./baseline/QEDG/sub_model_weight/resnet18_resnet34_svhn.pth",
    )
    p.add_argument(
        "--student_naive_posttrain_weight_path",
        type=str,
        default="./baseline/QEDG/mainexp/qedg_posttrain_weight/qedg_zeroquery_resnet18_resnet34_svhn.pth",
    )
    p.add_argument(
        "--student_addt_weight_path",
        type=str,
        default="./zero_query_weight_asr/qedg_addt_zeroquery_asr_resnet18_resnet34_svhn.pth",
    )

    p.add_argument("--figure_dpi", type=int, default=260)
    p.add_argument("--output_path", type=str, default="./agreement_svhn_qedg_4grid.png")
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
def _collect_preds(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = (y % 10).to(device, non_blocking=True)
        p = model(x).argmax(1)
        preds.append(p.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


def _confusion_matrix(a: np.ndarray, b: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(a.shape[0]):
        cm[int(a[i]), int(b[i])] += 1
    return cm


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1)
    return cm / denom


def _plot_heatmap(ax, mat: np.ndarray, title_main: str, title_sub: str, vmin: float, vmax: float):
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_title(
        title_main,
        fontsize=13,
        fontweight="bold",
        pad=12,
        y=1.06,
    )
    ax.text(
        0.5,
        1.01,
        title_sub,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
    )
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.tick_params(axis="both", which="both", labelsize=11, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main():
    args = _parse_args()
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataset(args.dataset)
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

    t_pred, y_true = _collect_preds(target, loader, device, args.max_batches)
    s0_pred, _ = _collect_preds(no_post, loader, device, args.max_batches)
    s1_pred, _ = _collect_preds(naive_post, loader, device, args.max_batches)
    s2_pred, _ = _collect_preds(addt, loader, device, args.max_batches)

    target_acc = float((t_pred == y_true).mean() * 100.0)

    def metrics(student_pred: np.ndarray):
        agree = float((student_pred == t_pred).mean() * 100.0)
        kappa = float(cohen_kappa_score(student_pred, t_pred) * 100.0)
        return agree, kappa

    s0_agree, s0_kappa = metrics(s0_pred)
    s1_agree, s1_kappa = metrics(s1_pred)
    s2_agree, s2_kappa = metrics(s2_pred)

    cm_target = _row_normalize(_confusion_matrix(y_true, t_pred, 10))
    cm_s0 = _row_normalize(_confusion_matrix(t_pred, s0_pred, 10))
    cm_s1 = _row_normalize(_confusion_matrix(t_pred, s1_pred, 10))
    cm_s2 = _row_normalize(_confusion_matrix(t_pred, s2_pred, 10))

    fig = plt.figure(figsize=(8.8, 9.8), dpi=args.figure_dpi, constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.075,
        wspace=0.006,
        hspace=0.50,
    )
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    axes = [ax00, ax01, ax10, ax11]

    im0 = _plot_heatmap(axes[0], cm_target, "Target vs GT", f"Acc: {target_acc:.2f}%", vmin=0.0, vmax=1.0)
    _plot_heatmap(
        axes[1],
        cm_s0,
        "No post-train vs Target",
        f"Agree: {s0_agree:.2f}%, Kappa: {s0_kappa:.2f}%",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_heatmap(
        axes[2],
        cm_s1,
        "Naive post-train vs Target",
        f"Agree: {s1_agree:.2f}%, Kappa: {s1_kappa:.2f}%",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_heatmap(
        axes[3],
        cm_s2,
        "ADDT vs Target",
        f"Agree: {s2_agree:.2f}%, Kappa: {s2_kappa:.2f}%",
        vmin=0.0,
        vmax=1.0,
    )

    for i, ax in enumerate(axes):
        ax.set_xlabel("Predicted Label", fontsize=12, labelpad=6)
        ax.set_ylabel("Reference Label", fontsize=12, labelpad=3)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
