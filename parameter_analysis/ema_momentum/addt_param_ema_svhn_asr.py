import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
from kornia import augmentation as K

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

from utils import (
    ImagePool,
    get_target_model,
    get_substitute_model,
    get_dataset,
    test_acc,
    test_cohen_kappa,
    test_robust,
    test_robust_with_eval_params,
    setup_seed,
)

warnings.filterwarnings("ignore")

TEACHER_MODEL_NAME = 'resnet34'
STUDENT_MODEL_NAME = 'resnet18'
DATASET = 'svhn'
TEACHER_WEIGHT_PATH = './target_model_weight/resnet34_svhn.pth'
STUDENT_WEIGHT_PATH = './baseline/QEDG/sub_model_weight/resnet18_resnet34_svhn.pth'
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


def fixmatch_consistency(logits_t_weak, logits_s_strong, conf_th=0.95, T=0.5):
    with torch.no_grad():
        probs_w = torch.softmax(logits_t_weak / T, dim=1)
        pmax, y_hat = probs_w.max(dim=1)
        mask = pmax.ge(conf_th)
    if mask.any():
        return F.cross_entropy(logits_s_strong[mask], y_hat[mask])
    else:
        return logits_s_strong.new_tensor(0.0)


def hard_pl_no_gating(logits_t, logits_s, tau_low=0.90, tau_high=0.98, T=1.0):
    B = logits_t.size(0)
    device = logits_t.device

    with torch.no_grad():
        probs = F.softmax(logits_t / T, dim=1)
        pmax, y_hat = probs.max(dim=1)
        w = (pmax - tau_low) / (tau_high - tau_low)
        w = w.clamp_(min=0.0, max=1.0)
        mask = torch.ones(B, dtype=torch.bool, device=device)

    loss_i = F.cross_entropy(logits_s[mask], y_hat[mask], reduction='none')
    loss = (loss_i * w[mask]).sum() / (w[mask].sum() + 1e-8)
    used_frac = mask.float().mean()
    avg_w = w[mask].mean()
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


def compute_class_weights_from_loader(data_loader, num_classes: int):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, labels in data_loader:
        labels = labels.view(-1)
        counts.index_add_(0, labels.cpu().long(), torch.ones_like(labels, dtype=torch.float64))
    eps = 1e-6
    inv = 1.0 / (counts + eps)
    inv = inv * (num_classes / inv.sum().clamp_min(eps))
    return inv.to(dtype=torch.float32)


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
    ema_decay,
    epochs=EPOCHS,
    lr=LR,
    momentum=MOMENTUM,
    device="cuda",
    class_weights=None,
):
    student = student.to(device).train()
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum)

    ema_teacher = EMATeacher(student, ema_decay=ema_decay)

    weak = K.RandomCrop(size=(32, 32), padding=4, p=1.0)
    strong = nn.Sequential(
        K.RandomCrop(size=(32, 32), padding=4, p=1.0),
        K.ColorJitter(0.3, 0.3, 0.3, 0.1, p=1.0),
    )

    use_mixup = True

    if class_weights is None:
        class_weights = torch.ones(NUMBER_CLASSES, dtype=torch.float32)
    class_weights = class_weights.to(device)

    gamma_hard_ce = 0.3

    best_asr = float('-inf')

    save_dir = "./parameter_analysis_results/ema_momentum"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"ema_{ema_decay}_{STUDENT_MODEL_NAME}_{TEACHER_MODEL_NAME}_{DATASET}.pth"
    save_path = os.path.join(save_dir, save_name)

    print(f"Start Training {DATASET} | EMA Decay: {ema_decay}")

    for epoch in range(epochs):
        epoch_total_loss_sum = None
        epoch_ce_loss_sum = None
        epoch_cons_loss_sum = None
        epoch_cons_used_frac_sum = None
        epoch_cons_avg_w_sum = None
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(syn_loader):
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            images_s = strong(images)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                if n_weak_views <= 1:
                    logits_t = ema_teacher(weak(images))
                else:
                    logits_t = ema_teacher(weak(images))
                    for _ in range(n_weak_views - 1):
                        logits_t.add_(ema_teacher(weak(images)))
                    logits_t.div_(float(n_weak_views))

            logits_s_strong = student(images_s)

            loss_cons, cons_used_frac, cons_avg_w = hard_pl_no_gating(
                logits_t,
                logits_s_strong,
                tau_low=tau_low,
                tau_high=tau_high,
                T=T_SHARP,
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

            fc = _get_classifier_fc(student)
            loss_graph = alpha_cons * loss_cons

            loss_graph.backward(retain_graph=True)
            if fc.weight.grad is not None:
                fc.weight.grad.zero_()
            if fc.bias.grad is not None:
                fc.bias.grad.zero_()
            loss_ce.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                ema_teacher.update(student)

            batch_total_loss = loss_ce.detach() + loss_graph.detach()
            if epoch_total_loss_sum is None:
                epoch_total_loss_sum = batch_total_loss.clone()
                epoch_ce_loss_sum = loss_ce.detach().clone()
                epoch_cons_loss_sum = loss_cons.detach().clone()
                epoch_cons_used_frac_sum = cons_used_frac.detach().clone()
                epoch_cons_avg_w_sum = cons_avg_w.detach().clone()
            else:
                epoch_total_loss_sum.add_(batch_total_loss)
                epoch_ce_loss_sum.add_(loss_ce.detach())
                epoch_cons_loss_sum.add_(loss_cons.detach())
                epoch_cons_used_frac_sum.add_(cons_used_frac.detach())
                epoch_cons_avg_w_sum.add_(cons_avg_w.detach())
            num_batches += 1

        avg_loss = (
            (epoch_total_loss_sum / num_batches).item()
            if num_batches > 0 and epoch_total_loss_sum is not None
            else 0.0
        )
        acc, _ = test_acc(student, test_loader)
        con = test_cohen_kappa(student, teacher, test_loader)
        asr = test_robust_with_eval_params(student, teacher, test_loader, 'FGSM', False, 'svhn')

        print("-" * 120)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Kappa: {con:.2f}% | ASR: {asr:.2f}%")

        current_asr = asr.item() if torch.is_tensor(asr) else asr
        if current_asr > best_asr:
            best_asr = current_asr
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(student.state_dict(), save_path)
            print(f"New best ASR ({best_asr:.2f}%) saved to {save_path}")


def load_teacher_model():
    if not os.path.exists(TEACHER_WEIGHT_PATH):
        raise FileNotFoundError(f"Teacher weights not found at {TEACHER_WEIGHT_PATH}")
    teacher_model = get_target_model(DATASET, TEACHER_MODEL_NAME)
    state_dict = torch.load(TEACHER_WEIGHT_PATH, map_location="cpu")
    teacher_model.load_state_dict(state_dict)
    return teacher_model


def load_student_model():
    if not os.path.exists(STUDENT_WEIGHT_PATH):
        raise FileNotFoundError(f"Student weights not found at {STUDENT_WEIGHT_PATH}")
    student_model = get_substitute_model(DATASET, STUDENT_MODEL_NAME)
    state_dict = torch.load(STUDENT_WEIGHT_PATH, map_location="cpu")
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
        persistent_workers=True,
    )
    return syn_dataset, syn_loader


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Post-train zero-query with configurable PL params")
    parser.add_argument("--tau_low", type=float, default=0.92, help="Lower confidence threshold for pseudo-label weighting")
    parser.add_argument("--tau_high", type=float, default=0.99, help="Upper confidence threshold for pseudo-label weighting")
    parser.add_argument("--n_weak_views", type=int, default=1, help="Number of weak views for teacher ensembling")
    parser.add_argument("--margin_th", type=float, default=0.25, help="Top1-Top2 margin threshold for selecting pseudo-labels")
    parser.add_argument("--alpha_cons", type=float, default=0.22, help="Weight for consistency loss term")
    parser.add_argument("--ema_decay", type=float, default=0.90, help="EMA decay")
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
        ema_decay=args.ema_decay,
    )


if __name__ == '__main__':
    main()
