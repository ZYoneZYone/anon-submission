#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import math
import gc
import sys
import time
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from nets import resnet18, resnet34, VGG16, VGG19, mobilenet_v2
from utils import (ImagePool, get_dataset, setup_seed, get_target_model,
                   get_substitute_model, test_acc, test_cohen_kappa, test_robust,
                   write, print_header, print_section, print_single_model_summary,
                   data_aug)

cudnn.benchmark = True


class Loss_max(nn.Module):
    
    def __init__(self, beta=0.1):
        super(Loss_max, self).__init__()
        self.beta = beta
        return

    def forward(self, pred, truth, proba):
        criterion_ce = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()
        pred_prob = F.softmax(pred, dim=1)
        loss = criterion_ce(pred, truth) + criterion_mse(pred_prob, proba) * self.beta
        final_loss = torch.exp(loss * -1)
        return final_loss


class pre_conv(nn.Module):
    def __init__(self, nz, nc, img_size):
        super(pre_conv, self).__init__()
        self.nf = 64
        self.nc = nc
        self.img_size = img_size
        self.pre_conv = nn.Sequential(
            nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        output = self.pre_conv(input)
        return output


class Generator(nn.Module):
    def __init__(self, nc):
        super(Generator, self).__init__()
        self.nf = 64
        self.nc = nc
        self.main = nn.Sequential(
            nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nf, self.nc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.nc, self.nc, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output


def chunks(arr, m):
    """Split tensor into m chunks for per-class generation."""
    n = int(math.ceil(arr.size(0) / float(m)))
    return [arr[i:i + n] for i in range(0, arr.size(0), n)]


def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ADDTSynthesizer():
    
    def __init__(self, nz, num_classes, img_size, nc, batch_size,
                 dataset, subnet_name, images_dir, args):
        super(ADDTSynthesizer, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.img_size = img_size
        self.nc = nc
        self.batch_size = batch_size
        self.dataset = dataset
        self.subnet_name = subnet_name
        self.args = args

        self.data_pool = ImagePool(images_dir, dataset, subnet_name)
        self.aug, self.transform = data_aug(dataset)

        self.pre_conv_block = []
        for i in range(num_classes):
            block = nn.DataParallel(pre_conv(nz, nc, img_size).cuda())
            block.apply(weights_init)
            self.pre_conv_block.append(block)

        self.netG = Generator(nc).cuda()
        self.netG.apply(weights_init)
        self.netG = nn.DataParallel(self.netG)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_max = Loss_max(beta=args.beta)

        self.optimizerG = optim.Adam(self.netG.parameters(),
                                      lr=args.lr_g, betas=(args.beta1, 0.999))
        self.optimizer_block = []
        for i in range(num_classes):
            opt = optim.Adam(self.pre_conv_block[i].parameters(),
                           lr=args.lr_g, betas=(args.beta1, 0.999))
            self.optimizer_block.append(opt)

    def get_data(self):
        """Retrieve all data from ImagePool."""
        datasets = self.data_pool.get_dataset()
        global query
        query = len(datasets)
        print_section(f"Query Update - Total Queries: {query}")
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        return self.data_loader

    def gen_data(self, sub_net, target_net):
        """Generate samples using dual-stage per-class generator."""
        sub_net.eval()
        target_net.eval()
        device = torch.device("cuda:0")

        noise = torch.randn(self.batch_size, self.nz, 1, 1, device=device).cuda()
        noise_chunk = chunks(noise, self.num_classes)

        data_list = []
        set_label_list = []
        for i in range(len(noise_chunk)):
            tmp_data = self.pre_conv_block[i](noise_chunk[i])
            gene_data = self.netG(tmp_data)
            label = torch.full((noise_chunk[i].size(0),), i).cuda()
            data_list.append(gene_data)
            set_label_list.append(label)

        data = torch.cat(data_list, 0)
        set_label = torch.cat(set_label_list, 0)
        index = torch.randperm(set_label.size()[0])
        data = data[index]
        set_label = set_label[index]

        if self.dataset in ['mnist', 'fmnist'] and data.shape[2] < 28:
            data = F.interpolate(data, size=(28, 28), mode='bilinear', align_corners=False)
        elif self.dataset in ['cifar10', 'svhn', 'cifar100'] and data.shape[2] < 32:
            data = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
        elif self.dataset == 'tiny-imagenet' and data.shape[2] < 64:
            data = F.interpolate(data, size=(64, 64), mode='bilinear', align_corners=False)

        data = self.aug(data)
        self.data_pool.add(data, self.transform, target_net, sub_net,
                          self.subnet_name, threshold=1)
        return data, set_label

    def gen_data_with_g_steps(self, sub_net, target_net, g_steps):
        """Generate data with internal optimization steps."""
        sub_net.eval()
        target_net.eval()
        device = torch.device("cuda:0")

        noise = torch.randn(self.batch_size, self.nz, 1, 1, device=device).cuda()
        noise.requires_grad = True
        noise_chunk = chunks(noise, self.num_classes)

        best_data = None
        best_loss = float('inf')
        optimizers = [self.optimizerG] + self.optimizer_block

        for step in range(g_steps):
            data_list = []
            set_label_list = []
            for i in range(len(noise_chunk)):
                tmp_data = self.pre_conv_block[i](noise_chunk[i])
                gene_data = self.netG(tmp_data)
                label = torch.full((noise_chunk[i].size(0),), i).cuda()
                data_list.append(gene_data)
                set_label_list.append(label)

            data = torch.cat(data_list, 0)
            set_label = torch.cat(set_label_list, 0)
            index = torch.randperm(set_label.size()[0])
            data = data[index]
            set_label = set_label[index]

            if self.dataset in ['mnist', 'fmnist'] and data.shape[2] < 28:
                data = F.interpolate(data, size=(28, 28), mode='bilinear', align_corners=False)
            elif self.dataset in ['cifar10', 'svhn', 'cifar100'] and data.shape[2] < 32:
                data = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
            elif self.dataset == 'tiny-imagenet' and data.shape[2] < 64:
                data = F.interpolate(data, size=(64, 64), mode='bilinear', align_corners=False)

            data_aug = self.aug(data)

            with torch.no_grad():
                outputs = target_net(data_aug)
                _, label = torch.max(outputs.data, 1)
                outputs = F.softmax(outputs, dim=1)

            output = sub_net(data_aug.detach())
            prob = F.softmax(output, dim=1)
            errD_prob = mse_loss(prob, outputs, reduction='mean')
            errD_fake = self.criterion(output, label) + errD_prob * self.args.beta

            if errD_fake.item() < best_loss:
                best_loss = errD_fake.item()
                best_data = data.data.clone()

            self.netG.zero_grad()
            for i in range(self.num_classes):
                self.pre_conv_block[i].zero_grad()
            errD_fake.backward()
            for optimizer in optimizers:
                optimizer.step()

        if self.dataset in ['mnist', 'fmnist'] and best_data.shape[2] < 28:
            best_data = F.interpolate(best_data, size=(28, 28), mode='bilinear', align_corners=False)
        best_data = self.aug(best_data)
        self.data_pool.add(best_data, self.transform, target_net, sub_net,
                          self.subnet_name, threshold=1)
        return best_data, set_label

    def train_step(self, sub_net, target_net):
        """Update substitute model using generated data."""
        sub_net.train()
        target_net.eval()

        data, set_label = self.gen_data(sub_net, target_net)

        sub_net.zero_grad()
        with torch.no_grad():
            outputs = target_net(data)
            _, label = torch.max(outputs.data, 1)
            outputs = F.softmax(outputs, dim=1)

        output = sub_net(data.detach())
        prob = F.softmax(output, dim=1)
        errD_prob = mse_loss(prob, outputs, reduction='mean')
        errD_fake = self.criterion(output, label) + errD_prob * self.args.beta
        errD_fake.backward()

        return errD_fake.item(), errD_prob.item(), data, set_label, label, outputs

    def generator_step(self, sub_net, data, set_label, label, outputs):
        """Update generator to maximize substitute model's error."""
        self.netG.zero_grad()
        for i in range(self.num_classes):
            self.pre_conv_block[i].zero_grad()

        output = sub_net(data)
        loss_imitate = self.criterion_max(pred=output, truth=label, proba=outputs)
        loss_diversity = self.criterion(output, set_label.squeeze().long())
        errG = self.args.alpha * loss_diversity + loss_imitate

        if loss_diversity.item() <= 0.1:
            self.args.alpha = loss_diversity.item()

        errG.backward()
        self.optimizerG.step()
        for i in range(self.num_classes):
            self.optimizer_block[i].step()

        return errG.item(), loss_imitate.item(), loss_diversity.item()


def addt_train(synthesizer, sub_net, target_net, optimizer_D):
    """Train substitute model on generated data from ImagePool."""
    sub_net.train()
    target_net.eval()

    data_loader = synthesizer.get_data()
    if len(data_loader) == 0:
        return

    total_loss = 0.0
    total = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer_D.zero_grad()
        outputs = sub_net(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer_D.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    print_section("Training on Generated Data")
    print(f"Average Loss: {avg_loss:.4f}")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                       help='dataset name')
    parser.add_argument('--black_net', type=str, default='resnet34',
                       help='target model architecture')
    parser.add_argument('--sub_net', type=str, default='resnet18',
                       help='substitute model architecture')
    parser.add_argument('--epochs', type=int, default=10,
                       help='number of epochs after query budget exhausted')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='learning rate for substitute model')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--lr_g', type=float, default=1e-3,
                       help='learning rate for generator')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='beta1 for Adam optimizer')
    parser.add_argument('--nz', type=int, default=1024,
                       help='dimension of noise vector')
    parser.add_argument('--g_steps', type=int, default=100,
                       help='number of generator steps per iteration')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='batch size for training')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='weight for diversity loss (adaptive)')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='weight for probability MSE loss')
    parser.add_argument('--G_type', type=int, default=1,
                       help='generator type (1 or 2)')
    parser.add_argument('--query', type=int, default=30000,
                       help='query budget')
    parser.add_argument('--sub_model_weight_path', type=str,
                       default='./sub_model_weight',
                       help='path to save substitute model weights')
    parser.add_argument('--save_images_dir', type=str,
                       default='./generated_images',
                       help='directory to save generated images')
    parser.add_argument('--seed', type=int, default=2021,
                       help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()

    print("====================================")
    print("The time now is:{}".format(time.localtime()))
    print('ADDT Method (Standalone)')
    print('Target model: {} Substitute model: {}'.format(
        args.black_net, args.sub_net))

    images_dir = args.save_images_dir
    subnet_tag = f"{args.sub_net}__{args.black_net}"
    model_specific_dir = os.path.join(images_dir, args.dataset, subnet_tag)
    if os.path.exists(model_specific_dir):
        shutil.rmtree(model_specific_dir)
    os.makedirs(os.path.join(model_specific_dir, "images"), exist_ok=True)
    open(os.path.join(model_specific_dir, "images_list.txt"), "w").close()
    open(os.path.join(model_specific_dir, "labels_list.txt"), "w").close()

    os.makedirs(args.sub_model_weight_path, exist_ok=True)
    sub_weight_file = os.path.join(
        args.sub_model_weight_path,
        f"{args.sub_net}_{args.black_net}_{args.dataset}.pth",
    )
    if os.path.isfile(sub_weight_file):
        os.remove(sub_weight_file)

    setup_seed(args.seed)
    _, test_loader = get_dataset(args.dataset)
    sub_net = get_substitute_model(args.dataset, args.sub_net)
    target_net = get_target_model(args.dataset, args.black_net)

    print_header("ADDT Attack Configuration")
    print(f"Target Model: {args.black_net}")
    print(f"Substitute Model: {args.sub_net}")
    print(f"Dataset: {args.dataset}")
    print(f"Query Budget: {args.query}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Alpha (diversity weight): {args.alpha}")
    print(f"Beta (prob MSE weight): {args.beta}")

    print_section("Black-Box Model Performance")
    acc, _ = test_acc(target_net, test_loader)
    print(f"Accuracy: {acc:.2f}%")
    print("Accuracy of the black-box model: {:.3f}%".format(acc))

    print_section("Initial Substitute Model Performance")
    acc, _ = test_acc(sub_net, test_loader)
    con = test_cohen_kappa(sub_net, target_net, test_loader)
    asr = test_robust(sub_net, target_net, test_loader)
    print_single_model_summary(acc, con, con, asr, args.sub_net,
                              query=0, Query=args.query,
                              best_acc=acc, best_asr=asr)
    write(acc, con, con, asr, args.dataset, best_acc=acc, best_asr=asr)

    if args.dataset in ['mnist', 'fmnist']:
        image_size = 28
        num_class = 10
        nc = 1
    elif args.dataset in ['cifar10', 'svhn']:
        image_size = 32
        num_class = 10
        nc = 3
    elif args.dataset == 'cifar100':
        image_size = 32
        num_class = 100
        nc = 3
    elif args.dataset == 'tiny-imagenet':
        image_size = 64
        num_class = 200
        nc = 3
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    synthesizer = ADDTSynthesizer(
        nz=args.nz,
        num_classes=num_class,
        img_size=image_size,
        nc=nc,
        batch_size=args.batch_size,
        dataset=args.dataset,
        subnet_name=subnet_tag,
        images_dir=images_dir,
        args=args
    )

    optimizer_D = optim.SGD(sub_net.parameters(), args.lr, args.momentum)

    best_con = 0.0
    best_acc = 0.0
    best_asr = 0.0
    global query
    query = 0
    Query = args.query

    print_header("ADDT Training Phase")

    iteration = 0
    while query < Query:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} - Query: {query}/{Query}")
        print(f"{'='*60}")

        for ii in range(args.g_steps):
            errD, errD_prob, data, set_label, label, outputs = \
                synthesizer.train_step(sub_net, target_net)
            errG, loss_imitate, loss_diversity = \
                synthesizer.generator_step(sub_net, data, set_label, label, outputs)
            if ii % 20 == 0:
                print(f'[{ii}/{args.g_steps}] '
                      f'D: {errD:.4f} D_prob: {errD_prob:.4f} '
                      f'G: {errG:.4f} Imitate: {loss_imitate:.4f} '
                      f'Diversity: {loss_diversity:.4f}')

        addt_train(synthesizer, sub_net, target_net, optimizer_D)

        acc, _ = test_acc(sub_net, test_loader)
        con = test_cohen_kappa(sub_net, target_net, test_loader)
        asr = test_robust(sub_net, target_net, test_loader)

        weight_path = os.path.join(
            args.sub_model_weight_path,
            f'{args.sub_net}_{args.black_net}_{args.dataset}.pth',
        )
        con_for_best = con if np.isfinite(con) else float("-inf")
        if iteration == 1:
            best_con = con_for_best
            torch.save(sub_net.state_dict(), weight_path)
        elif con_for_best > best_con:
            best_con = con_for_best
            torch.save(sub_net.state_dict(), weight_path)

        if acc > best_acc:
            best_acc = acc
        if asr > best_asr:
            best_asr = asr

        print_section(f"Results - Iteration {iteration}")
        print_single_model_summary(acc, con, best_con, asr, args.sub_net,
                                  query=query, Query=Query,
                                  best_acc=best_acc, best_asr=best_asr)
        write(acc, con, best_con, asr, args.dataset,
              best_acc=best_acc, best_asr=best_asr)

        del data, set_label
        torch.cuda.empty_cache()
        gc.collect()

    print_header("ADDT Training Completed")
    print(f"Best Cohen's Kappa: {best_con:.2f}%")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best ASR: {best_asr:.2f}%")
    print(f"Total Queries Used: {query}")