from __future__ import print_function

import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from advertorch.attacks import GradientSignAttack, PGDAttack, LinfPGDAttack
from advertorch.attacks import LinfBasicIterativeAttack

from utils import setup_seed, get_dataset, get_substitute_model, get_target_model, test_acc


def test_adver(net, tar_net, attack, targeted, testloader, dataset):
    if dataset in ["mnist", "fmnist"]:
        eps = 0.3
        alpha = 0.01
    elif dataset in ["svhn", "cifar10"]:
        eps = 8 / 255.0
        alpha = 2 / 255.0
    bim_iter = 120
    pgd_iter = 20

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    net.eval()
    tar_net.eval()

    if attack == 'BIM':
        adversary = LinfBasicIterativeAttack(
            net, loss_fn=loss_fn, eps=eps, nb_iter=bim_iter,
            eps_iter=alpha, clip_min=0.0, clip_max=1.0, targeted=targeted)
    elif attack == 'PGD':
        adversary = LinfPGDAttack(
            net, loss_fn=loss_fn, eps=eps, nb_iter=pgd_iter,
            eps_iter=alpha, clip_min=0.0, clip_max=1.0, targeted=targeted)
    elif attack == 'FGSM':
        adversary = GradientSignAttack(
            net, loss_fn=loss_fn, eps=eps, targeted=targeted)
    else:
        raise ValueError(f"Unknown attack: {attack}")

    correct = 0
    total = 0

    def _num_classes_from_dataset(name: str) -> int:
        return {"mnist": 10, "fmnist": 10, "cifar10": 10, "svhn": 10, "cifar100": 100, "tiny-imagenet": 200}[name]

    K = _num_classes_from_dataset(dataset)

    for inputs, y_true in testloader:
        inputs, y_true = inputs.cuda(), y_true.cuda()

        with torch.no_grad():
            pred_clean = tar_net(inputs).argmax(1)

        if targeted:
            idx = torch.where(pred_clean == y_true)[0]
            if idx.numel() == 0:
                continue
            t = (y_true + torch.randint(1, K, y_true.shape, device=y_true.device)) % K
            adv = adversary.perturb(inputs[idx], t[idx])
            with torch.no_grad():
                pred_adv = tar_net(adv).argmax(1)
            total += idx.numel()
            correct += (pred_adv == t[idx]).sum().item()
        else:
            idx = torch.where(pred_clean == y_true)[0]
            if idx.numel() == 0:
                continue
            adv = adversary.perturb(inputs[idx], y_true[idx])
            with torch.no_grad():
                pred_adv = tar_net(adv).argmax(1)
            total += idx.numel()
            correct += (pred_adv == y_true[idx]).sum().item()

    if total == 0:
        return float('nan')

    asr = 100.0 * (correct / total) if targeted else 100.0 - 100.0 * (correct / total)
    return asr


if __name__ == '__main__':
    setup_seed(2021)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--sub_model_path', type=str)
    parser.add_argument('--black_net', type=str, default='resnet34')
    parser.add_argument('--sub_net', type=str, default='resnet18')

    opt = parser.parse_args()
    cudnn.benchmark = True

    train_loader, test_loader = get_dataset(opt.dataset)
    val_loader = test_loader

    sub_net = get_substitute_model(opt.dataset, opt.sub_net)
    loaded = torch.load(opt.sub_model_path)
    state_dict = loaded['state_dict'] if isinstance(loaded, dict) and 'state_dict' in loaded else loaded
    sub_net.load_state_dict(state_dict)

    blackBox_net = get_target_model(opt.dataset, opt.black_net)

    acc, _ = test_acc(sub_net, val_loader)
    print("Accuracy of the sub_net:{:.3} % \n".format(acc))

    for attack in ['Untarget', 'Target']:
        for adv in ['FGSM', 'BIM', 'PGD']:
            asr = test_adver(sub_net, blackBox_net, adv, attack == 'Target', val_loader, opt.dataset)
            print(attack + " , " + "type: " + adv + ", ASR:{:.2f} %, ".format(asr))

