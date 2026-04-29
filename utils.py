import cv2
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from advertorch.attacks import LinfPGDAttack, LinfBasicIterativeAttack, GradientSignAttack
import os
import random
import json
from datetime import datetime
from torchvision import datasets, transforms
from nets import resnet18, resnet34, VGG16, VGG19, mobilenet_v2
from sklearn.metrics import cohen_kappa_score
from torchvision import transforms
from kornia import augmentation
import subprocess


def get_target_model(dataset, model_name):
    pretraind = './target_model_weight/'
    weight_name = model_name + '_' + dataset + '.pth'
    weight_path = pretraind + weight_name
    state_dict = torch.load(weight_path)
    if model_name == 'resnet34':
        model = resnet34(dataset).cuda()
        model.load_state_dict(state_dict)
    if model_name == 'VGG16':
        model = VGG16(dataset).cuda()
        model.load_state_dict(state_dict)
    if model_name == 'VGG19':
        model = VGG19(dataset).cuda()
        model.load_state_dict(state_dict)
    if model_name == 'resnet18':
        model = resnet18(dataset).cuda()
        model.load_state_dict(state_dict)
    return model


def get_advtrained_model(dataset, model_name):
    pretraind = './target_model_weight/adv_trained/'
    weight_name = model_name + '_' + dataset + '_adv.pth'
    weight_path = pretraind + weight_name
    state_dict = torch.load(weight_path)
    if model_name == 'resnet34':
        model = resnet34(dataset).cuda()
        model.load_state_dict(state_dict)
    if model_name == 'VGG16':
        model = VGG16(dataset).cuda()
        model.load_state_dict(state_dict)
    if model_name == 'VGG19':
        model = VGG19(dataset).cuda()
        model.load_state_dict(state_dict)
    if model_name == 'resnet18':
        model = resnet18(dataset).cuda()
        model.load_state_dict(state_dict)
    return model

def get_substitute_model(dataset, model_name):
    if model_name == 'resnet34':
        model = resnet34(dataset).cuda()
    if model_name == 'vgg16':
        model = VGG16(dataset).cuda()
    if model_name == 'vgg19':
        model = VGG19(dataset).cuda()
    if model_name == 'resnet18':
        model = resnet18(dataset).cuda()
    if model_name == 'mobilenet_v2':
        model = mobilenet_v2(dataset).cuda()
    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_acc(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    return acc, test_loss


def test_robust(sub_net, tar_net, test_loader, attack='FGSM', target=False):
    sub_net.eval()
    tar_net.eval()
    adversary = GradientSignAttack(sub_net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=32/255, targeted=target)
    correct = 0
    total = 0
    for each in test_loader:
        images = each[0].cuda()
        labels = each[1].cuda()
        idx = torch.argmax(tar_net(images), dim=1) == labels
        images = images[idx]
        labels = labels[idx]
        total += len(labels)
        adv_images = adversary.perturb(images, labels)
        predict = torch.argmax(tar_net(adv_images), dim=1)
        correct += (predict != labels).sum()
    return correct / total * 100.


def test_robust_with_eval_params(sub_net, tar_net, test_loader, attack='FGSM', targeted=False, dataset='cifar10'):
    if dataset in ["mnist", "fmnist"]:
        eps = 0.3
    elif dataset in ["svhn", "cifar10"]:
        eps = 8.0 / 255.0
    else:
        eps = 8.0 / 255.0

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    sub_net.eval()
    tar_net.eval()

    if attack == 'FGSM':
        adversary = GradientSignAttack(
            sub_net, loss_fn=loss_fn, eps=eps, targeted=targeted)
    else:
        raise ValueError(f"Only FGSM supported, got: {attack}")

    correct = 0
    total = 0

    for inputs, y_true in test_loader:
        inputs, y_true = inputs.cuda(), y_true.cuda()

        with torch.no_grad():
            pred_clean = tar_net(inputs).argmax(1)

        if targeted:
            idx = torch.where(pred_clean == y_true)[0]
            if idx.numel() == 0:
                continue

            K = 10
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

def write(acc, con, best_con, asr, dataset, best_acc=None, best_asr=None):
    print("Accuracy of the substitute model:{:.3} %".format(acc))
    if best_acc is not None:
        print("Best Accuracy of the substitute model:{:.3} %".format(best_acc))
    print("Consistency of substitute model and black box model:{:.3}%, best Consistency:{:.3} %".format(con,
                                                                                                        best_con))
    print("Attack success rate:{:.3} %".format(asr))
    if best_asr is not None:
        print("Best Attack success rate:{:.3} %".format(best_asr))


def test_cohen_kappa(s_net, t_net, test_loader):
    s_pred = []
    t_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            s_out = torch.softmax(s_net(data), dim=1)
            t_out = torch.softmax(t_net(data), dim=1)
            s_pred += torch.argmax(s_out, dim=1).tolist()
            t_pred += torch.argmax(t_out, dim=1).tolist()
    return cohen_kappa_score(s_pred, t_pred) * 100


def _download_file_with_curl(url, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "2", "-o", tmp_path, url], check=True)
    os.replace(tmp_path, dst_path)


def _ensure_svhn_split_downloaded(root, split):
    from torchvision.datasets.svhn import SVHN
    from torchvision.datasets.utils import check_integrity

    url, filename, md5 = SVHN.split_list[split]
    fpath = os.path.join(root, filename)
    if check_integrity(fpath, md5):
        return
    if os.path.exists(fpath):
        os.remove(fpath)
    _download_file_with_curl(url, fpath)
    if not check_integrity(fpath, md5):
        raise RuntimeError("SVHN file download failed or corrupted: {}".format(fpath))


def get_dataset(dataset):
    data_dir = './data/{}'.format(dataset)
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]),
                                       download=True)
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]), download=True)
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()])
                                              , download=True)
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ])
                                             , download=True)
    elif dataset == "svhn":
        _ensure_svhn_split_downloaded(data_dir, "train")
        _ensure_svhn_split_downloaded(data_dir, "test")
        train_dataset = datasets.SVHN(data_dir, split="train",
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]),
                                      download=False)
        test_dataset = datasets.SVHN(data_dir, split="test",
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]), download=False)
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_dir, train=True,
                                         transform=transforms.Compose(
                                             [
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                             ]),
                                         download=True)
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]),
                                        download=True)
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))
    elif dataset == "tiny":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }
        data_dir = "data/tiny-imagenet-200/"
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        train_dataset = image_datasets['train']
        test_dataset = image_datasets['val']
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                               shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=True, num_workers=0)

    return train_loader, test_loader


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def save_image_batch(imgs, root, _idx, transform, blackBox_net, student, subnet_name, dataset, threshold, labels=None):
    was_training = student.training if student is not None else False

    for idx in range(imgs.shape[0]):
        img = imgs[idx]
        if dataset == 'mnist' or dataset == 'fmnist':
            img = (transform(img).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')[0]
        else:
            img = (transform(img)[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        img_path = os.path.join(root, dataset, subnet_name, 'images')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        img_path = os.path.join(img_path, str(_idx) + '-{}.png'.format(idx))
        cv2.imwrite(img_path, img)
        if dataset == 'mnist' or dataset == 'fmnist':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            temp_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda().float() / 255
        else:
            temp_img = torch.from_numpy(cv2.imread(img_path)).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255

        should_save = True
        if student is not None:
            student.eval()
            with torch.no_grad():
                s_out = torch.max(torch.softmax(student(temp_img), dim=1), dim=1)[0]
            if was_training:
                student.train()
            should_save = (s_out <= threshold)
        else:
            should_save = (threshold >= 1.0)

        if should_save:
            subnet_dir = os.path.join(root, dataset, subnet_name)
            if not os.path.exists(subnet_dir):
                os.makedirs(subnet_dir)
            with open(os.path.join(root, dataset, subnet_name, 'images_list.txt'), 'a') as f:
                f.write(str(_idx) + '-{}.png'.format(idx) + '\n')
            if labels is None:
                label = torch.argmax(blackBox_net(temp_img), dim=1)
            else:
                label = labels[idx].detach()
                if label.numel() != 1:
                    label = label.reshape(-1)[0]
            with open(os.path.join(root, dataset, subnet_name, 'labels_list.txt'), 'a') as f:
                f.write(str(int(label)) + '\n')


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset,subnet_name):
        self.root = os.path.abspath(root)
        self.dataset = dataset
        self.subnet_name = subnet_name
        with open(os.path.join(root, dataset, subnet_name, 'images_list.txt'), 'r') as f:
            self.images_names = [line.strip() for line in f.readlines() if line.strip()]
        with open(os.path.join(root, dataset, subnet_name, 'labels_list.txt'), 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines() if line.strip()]
        if len(self.images_names) != len(self.labels):
            raise ValueError(f"Images and labels count mismatch: {len(self.images_names)} vs {len(self.labels)}")

    def __getitem__(self, idx):
        image_name = self.images_names[idx].strip()
        image_path = os.path.join(self.root, self.dataset, self.subnet_name, 'images', image_name)
        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            img = torch.from_numpy(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0) / 255
        else:
            img = torch.from_numpy(cv2.imread(image_path)).permute(2, 0, 1) / 255
        label = torch.tensor(self.labels[idx])
        return img, label

    def __len__(self):
        return len(self.images_names)


class ImagePool(object):
    def __init__(self, root, dataset,subnet_name):
        self.root = root
        self.dataset = dataset
        self._idx = 0
        self.subnet_name = subnet_name

    def add(self, imgs, transform, blackBox_net, student,subnet_name, threshold, labels=None):
        save_image_batch(imgs, self.root, self._idx, transform, blackBox_net, student,subnet_name, self.dataset, threshold, labels=labels)
        self._idx += 1

    def get_dataset(self):
        return build_image_pool_dataset(self.root, self.dataset, self.subnet_name)


def build_image_pool_dataset(root, dataset, subnet_name):
    return UnlabeledImageDataset(root, dataset, subnet_name)


def hard_label_to_one_hot(hard_label):
    input_dim = len(hard_label)
    one_hot = torch.zeros((input_dim, 10)).cuda()
    one_hot.scatter_(1, hard_label.unsqueeze(1), 1)
    return one_hot


def cosine_similarity(matrix):
    dot_product = torch.matmul(matrix, matrix.t())
    norm = torch.norm(matrix, dim=1, keepdim=True)
    similarity = dot_product / torch.matmul(norm, norm.t())
    return similarity


def sum_intra_similarity(matrix):
    similarity_matrix = cosine_similarity(matrix)
    mask = torch.triu(torch.ones(similarity_matrix.shape), diagonal=1).cuda()
    intra_similarity_sum = torch.sum(similarity_matrix * mask)
    return intra_similarity_sum


def diversity_loss(inputs, targets):
    similarity = 0.0
    inputs = inputs.reshape(inputs.shape[0], -1)
    for i in range(torch.max(targets) + 1):
        temp_inputs = inputs[targets == i]
        similarity += sum_intra_similarity(temp_inputs)
    return similarity / inputs.shape[0]

def over_confidence_loss(s_prob):
    batch_size = s_prob.shape[0]
    return torch.std(s_prob, dim=1).sum() / batch_size

def data_aug(dataset):
    if dataset == "fmnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation([90,180]),
            transforms.RandomCrop(size=(28, 28), padding=4),
        ])
        return transform,transform
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.RandomCrop(size=(28, 28), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation([90,180]),
        ])
        return transform,transform
    if dataset == 'svhn':
        transform = transforms.Compose(
            [augmentation.ColorJitter(0.2, 0.2, 0.2), augmentation.RandomChannelShuffle(p=0.5),
             augmentation.RandomGaussianNoise(p=0.2),
             augmentation.RandomSolarize(p=0.3),
             augmentation.RandomErasing(p=0.2),
             augmentation.RandomAffine(padding_mode='border', degrees=45),
             augmentation.RandomHorizontalFlip(),
             augmentation.RandomVerticalFlip(),
             augmentation.RandomPerspective(),
             augmentation.RandomInvert(p=0.2),
             ])
        aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(hue=0.3),
                transforms.RandomCrop(size=(32, 32), padding=4),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(90, expand=True) if torch.rand(1) < 0.3 else transforms.RandomRotation(0)
            ])
        return aug, transform
    if dataset == 'cifar10':
        transform = transforms.Compose(
            [augmentation.ColorJitter(0.2, 0.2, 0.2), augmentation.RandomChannelShuffle(p=0.5),
             augmentation.RandomGaussianNoise(p=0.2),
             augmentation.RandomSolarize(p=0.3),
             augmentation.RandomErasing(p=0.2),
             augmentation.RandomAffine(padding_mode='border', degrees=45),
             augmentation.RandomHorizontalFlip(),
             augmentation.RandomVerticalFlip(),
             augmentation.RandomPerspective(),
             augmentation.RandomInvert(p=0.2),
             augmentation.RandomCrop(size=(32, 32), padding=4),
             ])
        aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(hue=0.3),
                transforms.RandomCrop(size=(32, 32), padding=4),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(90, expand=True) if torch.rand(1) < 0.3 else transforms.RandomRotation(0),
            ])
        return aug, transform


def free_aug(data,substitute_outputs):
    substitute_prob = torch.max(torch.softmax(substitute_outputs, dim=1),dim=1)[0]
    if random.random()<0.05:
        random_num = random.randint(0,1)
        if random_num==0:
            data_ = transforms.RandomHorizontalFlip()(data)
        elif random_num==1:
            data_ = transforms.RandomVerticalFlip()(data)
        data[substitute_prob>0.95] = data_[substitute_prob>0.95]
    return data

def Imbalance_CrossEntropyLoss(substitute_outputs, labels,balance_scale):
    pred_labels = torch.max(substitute_outputs,dim=1)[1]
    right_loss = torch.nn.CrossEntropyLoss(reduction="sum")(substitute_outputs[pred_labels==labels], labels[pred_labels==labels])
    wrong_loss = torch.nn.CrossEntropyLoss(reduction="sum")(substitute_outputs[pred_labels!=labels], labels[pred_labels!=labels])
    return (right_loss+ balance_scale * wrong_loss)/substitute_outputs.shape[0]


def print_header(title, width=80):
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def print_section(title, width=60):
    print(f"\n{'─' * width}")
    print(f"\U0001F4CA {title}")
    print(f"{'─' * width}")


def print_metrics_table(metrics_data, title="Performance Metrics"):
    print(f"\n┌─ {title} " + "─" * (50 - len(title)))
    for key, value in metrics_data.items():
        if isinstance(value, (int, float)):
            print(f"│ {key:<25} : {value:>8.2f}%")
        else:
            print(f"│ {key:<25} : {value:>10}")
    print("└" + "─" * 50)


def print_single_model_summary(acc, con, best_con, asr, model_name, epoch_info="",
                               best_acc=None, best_asr=None, query=0, Query=0):
    print(f"\nSingle Model Results {epoch_info}")
    print(f"Queries Used: {query}/{Query}")
    acc_data = {
        f"Current {model_name} Accuracy": acc,
        f"Best {model_name} Accuracy": best_acc if best_acc is not None else acc
    }
    print_metrics_table(acc_data, "Accuracy Performance")

    con_data = {
        f"Current {model_name} Cohen's Kappa": con,
        f"Best {model_name} Cohen's Kappa": best_con
    }
    print_metrics_table(con_data, "Cohen's Kappa Performance")

    asr_data = {
        f"Current {model_name} ASR": asr,
        f"Best {model_name} ASR": best_asr if best_asr is not None else asr
    }
    print_metrics_table(asr_data, "ASR Performance")


def prepare_log(args, dual_model_training):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if dual_model_training:
        log_dir = os.path.join('train_log', args.dataset, f"{args.sub_net_1}_{args.sub_net_2}", timestamp)
    else:
        log_dir = os.path.join('train_log', args.dataset, args.sub_net, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'train.json')

    log_data = {
        "experiment_config": vars(args),
        "initial_metrics": {},
        "training_history": [],
        "final_metrics": {}
    }
    return log_dir, log_file_path, log_data


def save_log(log_file_path, log_data):
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=4)

class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)

class ScoreLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(ScoreLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)
            logits = logits.transpose(1, 2)
            logits = logits.contiguous().view(-1, logits.size(2))
        target = target.view(-1, 1)

        score = F.log_softmax(logits, 1)
        score = score.gather(1, target)
        loss = -1 * score

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def cal_label(black_net, data):
    with torch.no_grad():
        outputs = black_net(data.detach())
        _, label = torch.max(outputs.data, 1)
    label = label.detach().cpu().numpy()
    label = torch.from_numpy(label).cuda().long()
    return label

