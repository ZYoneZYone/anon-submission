# Query-Efficient Data-Free Black-Box Adversarial Attacks via Augmented Dual-Decoupled Training

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ with CUDA 11.8+ recommended.

## Pretrained Target Models

Pre-trained ResNet-34 target models are provided in `./target_model_weight/`:

| Dataset  | File |
|----------|------|
| MNIST    | `target_model_weight/resnet34_mnist.pth` |
| FMNIST   | `target_model_weight/resnet34_fmnist.pth` |
| SVHN     | `target_model_weight/resnet34_svhn.pth` |
| CIFAR-10 | `target_model_weight/resnet34_cifar10.pth` |

## Quick Start

**Standalone** (generator → synthetic pool → HADR → DGDT → ASR):

```bash
# MNIST (20K queries)
python main_results/standalone/addt_main.py --dataset mnist   --query 20000

# FMNIST (20K queries)
python main_results/standalone/addt_main.py --dataset fmnist  --query 20000

# SVHN (30K queries)
python main_results/standalone/addt_main.py --dataset svhn    --query 30000

# CIFAR-10 (30K queries)
python main_results/standalone/addt_main.py --dataset cifar10 --query 30000
```

**Plug-in** (QEDG pre-trained substitute + synthetic pool → HADR + DGDT post-training).

Plug-in mode requires first running QEDG's original implementation (generator training + query-based pool construction + substitute pre-training) to produce the substitute model and synthetic image pool. Place the resulting artifacts under `./baseline/QEDG/`, then run:

```bash
# MNIST (20K queries)
python main_results/plugin/qedg/addt_qedg_posttrain_zeroquery_mnist_asr.py

# FMNIST (20K queries)
python main_results/plugin/qedg/addt_qedg_posttrain_zeroquery_fmnist_asr.py

# SVHN (30K queries)
python main_results/plugin/qedg/addt_qedg_posttrain_zeroquery_svhn_asr.py

# CIFAR-10 (30K queries)
python main_results/plugin/qedg/addt_qedg_posttrain_zeroquery_cifar10_asr.py
```

## Experiments

All scripts follow a unified CLI: `--dataset {mnist|fmnist|svhn|cifar10} --target_arch {resnet34|resnet18}`.

| Experiment | Directory |
|-----------|-----------|
| Main results | `main_results/` |
| Component contributions | `ablation_analysis/component_contributions/` |
| Multi-view decomposition | `ablation_analysis/multiview_decomposition/` |
| Gradient decay factor | `parameter_analysis/gradient_decay/` |
| EMA momentum | `parameter_analysis/ema_momentum/` |
| Target model approximation | `target_model_approximation/` |

## Baseline Data

Plug-in mode requires each baseline's pre-trained substitute model and synthetic pool placed under `./baseline/{BASELINE}/`. Example for QEDG:

```
baseline/QEDG/
├── sub_model_weight/
│   └── resnet18_resnet34_{dataset}.pth
└── images_generated/
    └── {dataset}/
```

## Directory Structure

```
opensource/
├── utils.py / nets.py / eval.py
├── main_results/
│   ├── standalone/
│   └── plugin/{dast,dfhl,dfta,ideal,mcnet,qedg,stv2}/
├── ablation_analysis/
│   ├── component_contributions/
│   ├── multiview_decomposition/
├── parameter_analysis/
│   ├── gradient_decay/
│   └── ema_momentum/
└── target_model_approximation/
    ├── agreement/
    └── js_divergence/
```
