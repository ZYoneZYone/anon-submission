import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
import torchvision.transforms as T

from utils import (
    ImagePool,
    get_target_model,
    get_substitute_model,
    get_dataset,
    test_acc,
    test_cohen_kappa,
    test_robust_with_eval_params,
    setup_seed,
)

warnings.filterwarnings("ignore")

TEACHER_MODEL_NAME = 'resnet34'
STUDENT_MODEL_NAME = 'resnet18'
DATASET = 'cifar10'

TEACHER_WEIGHT_PATH = './target_model_weight/resnet34_cifar10.pth'
STUDENT_WEIGHT_PATH = './baseline/QEDG/sub_model_weight/resnet18_resnet34_cifar10.pth'
SYNTHETIC_DATASET_PATH = './baseline/QEDG/images_generated'

EPOCHS = 30
LR = 0.01
MOMENTUM = 0.9
NUMBER_CLASSES = 10

T_SHARP = 1.00


def _get_classifier_fc(model: nn.Module) -> nn.Linear:
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        return model.classifier
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise ValueError("No linear layer found as classifier head.")
    return last_linear


class EMATeacher(nn.Module):
    def __init__(self, student, ema_decay=0.99):
        super().__init__()
        import copy
        self.teacher = copy.deepcopy(student).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update(self, student):
        d = self.ema_decay
        for ps, pt in zip(student.parameters(), self.teacher.parameters()):
            pt.mul_(d).add_(ps, alpha=1 - d)
        for (_, buf_s), (_, buf_t) in zip(student.named_buffers(), self.teacher.named_buffers()):
            if buf_t.dtype.is_floating_point:
                buf_t.mul_(d).add_(buf_s, alpha=1 - d)
            else:
                buf_t.copy_(buf_s)

    @torch.no_grad()
    def forward(self, x):
        return self.teacher(x)


def hard_pl_with_conf_weight(logits_t, logits_s,
                             tau_low=0.90, tau_high=0.98, T=1.0,
                             margin_th=0.20, top_frac=None):
    with torch.no_grad():
        probs = F.softmax(logits_t / T, dim=1)
        pmax, y_hat = probs.max(dim=1)

        w = (pmax - tau_low) / (tau_high - tau_low)
        w = w.clamp_(min=0.0, max=1.0)

        mask = torch.ones_like(pmax, dtype=torch.bool)

    if not mask.any():
        zero = (logits_s * 0.0).sum()
        return zero, 0.0, 0.0

    loss_i = F.cross_entropy(logits_s[mask], y_hat[mask], reduction='none')
    loss = (loss_i * w[mask]).sum() / (w[mask].sum() + 1e-8)
    used_frac = float(mask.float().mean().item())
    avg_w = float(w[mask].mean().item())
    return loss, used_frac, avg_w


def mixup_batch(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    y_onehot = F.one_hot(y, num_classes=NUMBER_CLASSES).float()
    y_mix = lam * y_onehot + (1.0 - lam) * y_onehot[idx]
    return x_mix, y_mix


def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_batch(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, C, H, W = x.size()
    idx = torch.randperm(B, device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    x1, y1, x2b, y2b = rand_bbox(W, H, lam)
    x_mix = x.clone()
    x_mix[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]
    lam_adj = 1.0 - ((x2b - x1) * (y2b - y1) / (W * H))
    y_one = F.one_hot(y, num_classes=NUMBER_CLASSES).float()
    y_two = F.one_hot(y2, num_classes=NUMBER_CLASSES).float()
    y_mix = lam_adj * y_one + (1.0 - lam_adj) * y_two
    return x_mix, y_mix


def soft_ce(logits, y_soft):
    logp = F.log_softmax(logits, dim=1)
    return -(y_soft * logp).sum(dim=1).mean()


def apply_on_batch(transform, x):
    return torch.stack([transform(img) for img in x], dim=0)


def compute_class_weights_from_loader(data_loader, num_classes: int):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, labels in data_loader:
        labels = labels.view(-1)
        counts.index_add_(0, labels.cpu().long(), torch.ones_like(labels, dtype=torch.float64))
    eps = 1e-6
    inv = 1.0 / (counts + eps)
    inv = inv * (num_classes / inv.sum().clamp_min(eps))
    return inv.to(dtype=torch.float32)


def grab_pre_fc_feats_once(model, images):
    feats_buf = {}
    fc = _get_classifier_fc(model)

    def pre_fc_hook(module, inputs):
        feats_buf['feats'] = inputs[0]

    handle = fc.register_forward_pre_hook(pre_fc_hook)
    logits = model(images)
    handle.remove()
    feats = feats_buf.pop('feats')
    return logits, feats


def zero_query_post_train(
    syn_loader,
    student,
    test_loader,
    teacher,
    tau_low,
    tau_high,
    n_weak_views,
    margin_th,
    alpha_cons,
    decay_factor,
    epochs=EPOCHS,
    lr=LR,
    momentum=MOMENTUM,
    device="cuda",
    class_weights=None,
):
    student = student.to(device).train()
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum)

    ema_teacher = EMATeacher(student, ema_decay=0.995)

    weak = T.Compose([
        T.RandomCrop(32, padding=4),
    ])
    strong = T.Compose([
        T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=0),
    ])

    use_mixup = True

    if class_weights is None:
        class_weights = torch.ones(NUMBER_CLASSES, dtype=torch.float32)
    class_weights = class_weights.to(device)

    gamma_hard_ce = 0.3

    best_asr = float('-inf')
    save_dir = "./ablation_results/gradient_decay"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"addt_param_decay_{decay_factor}_{STUDENT_MODEL_NAME}_{TEACHER_MODEL_NAME}_{DATASET}.pth"
    save_path = os.path.join(save_dir, save_name)

    for epoch in range(epochs):
        epoch_total_loss_sum = 0.0
        epoch_ce_loss_sum = 0.0
        epoch_cons_loss_sum = 0.0
        epoch_cons_used_frac_sum = 0.0
        epoch_cons_avg_w_sum = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(syn_loader):
            images_s = apply_on_batch(strong, images).to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                logits_list = []
                for _ in range(n_weak_views):
                    images_w_i = apply_on_batch(weak, images).to(device).float()
                    logits_list.append(ema_teacher(images_w_i))
                logits_t = torch.stack(logits_list, dim=0).mean(0)

            logits_s_strong, feats_strong = grab_pre_fc_feats_once(student, images_s)

            loss_cons, cons_used_frac, cons_avg_w = hard_pl_with_conf_weight(
                logits_t,
                logits_s_strong,
                tau_low=tau_low,
                tau_high=tau_high,
                T=T_SHARP,
                margin_th=margin_th,
                top_frac=None,
            )

            if use_mixup:
                if np.random.rand() < 0.5:
                    images_mix, y_soft = mixup_batch(images_s, labels, alpha=0.4)
                else:
                    images_mix, y_soft = cutmix_batch(images_s, labels, alpha=1.0)
                logits_mix = student(images_mix)
                loss_ce_mix = soft_ce(logits_mix, y_soft)
            else:
                batch_size = labels.size(0)
                loss_ce_mix = F.cross_entropy(logits_s_strong, labels, reduction="sum") / batch_size

            ce_hard_balanced = F.cross_entropy(logits_s_strong, labels, weight=class_weights)
            loss_ce = loss_ce_mix + gamma_hard_ce * ce_hard_balanced

            total_loss = loss_ce + alpha_cons * loss_cons

            fc = _get_classifier_fc(student)
            loss_graph = alpha_cons * loss_cons

            if cons_used_frac > 0:
                loss_graph.backward(retain_graph=True)
                if fc.weight.grad is not None:
                    fc.weight.grad.mul_(decay_factor)
                if fc.bias.grad is not None:
                    fc.bias.grad.mul_(decay_factor)

            loss_ce.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                ema_teacher.update(student)

            epoch_total_loss_sum += total_loss.item()
            epoch_ce_loss_sum += loss_ce.item()
            epoch_cons_loss_sum += loss_cons.item()
            epoch_cons_used_frac_sum += cons_used_frac
            epoch_cons_avg_w_sum += cons_avg_w
            num_batches += 1

        avg_loss = epoch_total_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_loss_ce = epoch_ce_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_cons = epoch_cons_loss_sum / num_batches if num_batches > 0 else 0.0

        acc, _ = test_acc(student, test_loader)
        con = test_cohen_kappa(student, teacher, test_loader)
        asr = test_robust_with_eval_params(student, teacher, test_loader, 'FGSM', False, DATASET)

        print("-" * 120)
        print(f"[Gradient Decay {decay_factor}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | CE: {avg_loss_ce:.4f} | Cons: {avg_cons:.4f}")
        print(f"Eval - Acc: {acc:.2f}% | Cohen's Kappa: {con:.2f}% | ASR: {asr:.2f}%")
        cons_used_frac = epoch_cons_used_frac_sum / num_batches if num_batches > 0 else 0.0
        cons_avg_w = epoch_cons_avg_w_sum / num_batches if num_batches > 0 else 0.0
        print(f"Cons-used: {cons_used_frac:.3f} | Cons-avgW: {cons_avg_w:.3f}")

        current_asr = asr.item() if torch.is_tensor(asr) else asr
        if current_asr > best_asr:
            best_asr = current_asr
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(student.state_dict(), save_path)
            print(f"[Epoch {epoch+1}] New best ASR ({current_asr:.2f}%) saved to {save_path}")


def load_teacher_model():
    if not os.path.exists(TEACHER_WEIGHT_PATH):
        raise FileNotFoundError(f"Teacher weights not found at {TEACHER_WEIGHT_PATH}")
    teacher_model = get_target_model(DATASET, TEACHER_MODEL_NAME)
    return teacher_model


def load_student_model():
    if not os.path.exists(STUDENT_WEIGHT_PATH):
        raise FileNotFoundError(f"Student weights not found at {STUDENT_WEIGHT_PATH}")
    student_model = get_substitute_model(DATASET, STUDENT_MODEL_NAME)
    state_dict = torch.load(STUDENT_WEIGHT_PATH)
    student_model.load_state_dict(state_dict)
    return student_model


def load_synthetic_dataset_and_loader():
    root = SYNTHETIC_DATASET_PATH
    dataset = DATASET
    subnet_name = STUDENT_MODEL_NAME

    expected_dir = os.path.join(root, dataset, subnet_name)
    if not os.path.isdir(expected_dir):
        raise FileNotFoundError(f"Synthetic pool directory not found: {expected_dir}")

    image_pool = ImagePool(root, dataset, subnet_name)
    syn_dataset = image_pool.get_dataset()
    syn_loader = torch.utils.data.DataLoader(
        syn_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    return syn_dataset, syn_loader


def build_arg_parser():
    parser = argparse.ArgumentParser(description="ADDT Parameter Analysis: Gradient Decay Factor — CIFAR10")
    parser.add_argument("--tau_low", type=float, default=0.88, help="Lower confidence threshold")
    parser.add_argument("--tau_high", type=float, default=0.985, help="Upper confidence threshold")
    parser.add_argument("--n_weak_views", type=int, default=2, help="Number of weak views")
    parser.add_argument("--margin_th", type=float, default=0.12, help="Margin threshold")
    parser.add_argument("--alpha_cons", type=float, default=0.25, help="Consistency loss weight")
    parser.add_argument("--decay_factor", type=float, required=True,
                        help="Gradient decay factor for classifier head consistency gradients (0 = full decoupling, 1 = no decay)")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_seed(2021)
    _, test_loader = get_dataset(DATASET)

    teacher = load_teacher_model()
    student = load_student_model()

    teacher_acc, _ = test_acc(teacher, test_loader)
    student_acc, _ = test_acc(student, test_loader)

    print(f"Teacher ({TEACHER_MODEL_NAME}, {DATASET}) Accuracy: {teacher_acc:.2f}%")
    print(f"Student ({STUDENT_MODEL_NAME}, {DATASET}) Accuracy: {student_acc:.2f}%")

    syn_dataset, syn_loader = load_synthetic_dataset_and_loader()
    tmp_loader = torch.utils.data.DataLoader(
        syn_dataset, batch_size=512, shuffle=False, num_workers=2, drop_last=False
    )
    class_weights = compute_class_weights_from_loader(tmp_loader, NUMBER_CLASSES)
    print(f"Synthetic pool loaded ({DATASET}/{STUDENT_MODEL_NAME}): {len(syn_dataset)} samples")

    print("=" * 120)
    print("ADDT PARAMETER ANALYSIS: Gradient Decay Factor")
    print(f"Decay factor = {args.decay_factor} (0 = full decoupling, 1 = no decay)")
    print("=" * 120)

    zero_query_post_train(
        syn_loader,
        student,
        test_loader,
        teacher,
        class_weights=class_weights,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
        n_weak_views=args.n_weak_views,
        margin_th=args.margin_th,
        alpha_cons=args.alpha_cons,
        decay_factor=args.decay_factor,
    )


if __name__ == '__main__':
    main()
