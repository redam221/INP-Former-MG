"""
Training / evaluation entry point for INP-Former-MG.

Reproduces the original INP-Former training loop, but with configurable
granularity: ``--layer-groups "0,1,2,3,4,5,6,7"`` (single group) recovers the
original model, while e.g. ``--layer-groups "0,1,2;3,4,5;6,7"`` builds a
3-granularity variant.

Examples
--------
# single-granularity baseline (== original INP-Former)
python train.py --dataset MVTec-AD --data-path ../mvtec_anomaly_detection \
    --layer-groups "0,1,2,3,4,5,6,7" --inp-nums 6 --phase train

# 3-granularity variant (shallow / middle / deep), soft coherence + diversity
python train.py --dataset MVTec-AD --data-path ../mvtec_anomaly_detection \
    --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 \
    --coherence-mode soft --diversity-weight 0.05 --phase train
"""

from __future__ import annotations

import argparse
import os
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dataset import MVTecDataset, RealIADDataset, get_data_transforms
from utils import (
    evaluation_batch,
    WarmCosineScheduler,
    global_cosine_hm_adaptive,
    setup_seed,
    get_logger,
)
from optimizers import StableAdamW
from models.builder import build_model

warnings.filterwarnings("ignore")


def build_dataloaders(args):
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    train_list, test_list = [], []

    if args.dataset in ("MVTec-AD", "VisA"):
        for i, item in enumerate(args.item_list):
            train_path = os.path.join(args.data_path, item, "train")
            test_path = os.path.join(args.data_path, item)
            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(s[0], i) for s in train_data.samples]
            test_data = MVTecDataset(root=test_path, transform=data_transform,
                                     gt_transform=gt_transform, phase="test")
            train_list.append(train_data)
            test_list.append(test_data)
    elif args.dataset == "Real-IAD":
        for i, item in enumerate(args.item_list):
            train_data = RealIADDataset(root=args.data_path, category=item,
                                        transform=data_transform,
                                        gt_transform=gt_transform, phase="train")
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            test_data = RealIADDataset(root=args.data_path, category=item,
                                       transform=data_transform,
                                       gt_transform=gt_transform, phase="test")
            train_list.append(train_data)
            test_list.append(test_data)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_data = ConcatDataset(train_list)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    return train_loader, test_list


def _print_metrics(item, results, print_fn):
    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
    print_fn(f'{item}: I-Auroc:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, '
             f'P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, '
             f'P-AUPRO:{aupro_px:.4f}')


def main(args, print_fn=print):
    setup_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader, test_list = build_dataloaders(args)

    model = build_model(
        encoder_name=args.encoder,
        layer_groups=args.layer_groups,
        inp_nums=args.inp_nums,
        decoder_layers=args.decoder_layers or None,
        remove_class_token=True,
        coherence_mode=args.coherence_mode,
        coherence_temperature=args.coherence_temperature,
        diversity_weight=args.diversity_weight,
        adaptive_prototypes=args.adaptive_prototypes,
        gate_temperature=args.gate_temperature,
        gate_sparsity_weight=args.gate_sparsity_weight,
    ).to(device)

    # Collect the trainable parts (everything except the frozen encoder).
    trainable = [model.groups, model.prototype_tokens]
    if model.adaptive_prototypes:
        trainable.append(model.prototype_logits)
    trainable = nn.ModuleList(trainable)

    if args.phase == "train":
        # Parameter init (mirrors the original INP-Former).
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW([{"params": trainable.parameters()}],
                                lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4,
                                amsgrad=True, eps=1e-10)
        scheduler = WarmCosineScheduler(
            optimizer, base_value=args.lr, final_value=args.lr * 0.1,
            total_iters=args.total_epochs * len(train_loader), warmup_iters=100)

        print_fn(f"train image number: {len(train_loader.dataset)}")
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_loader, ncols=80):
                img = img.to(device)
                en, de, coherence_loss = model(img)
                recon_loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = recon_loss + args.coherence_weight * coherence_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                scheduler.step()
            print_fn(f"epoch [{epoch + 1}/{args.total_epochs}], loss: {np.mean(loss_list):.4f}")

            if (epoch + 1) % args.total_epochs == 0:
                _evaluate(model, test_list, args, device, print_fn)
                os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, args.save_name, "model.pth"))
                model.train()

    elif args.phase == "test":
        ckpt = os.path.join(args.save_dir, args.save_name, "model.pth")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        model.eval()
        _evaluate(model, test_list, args, device, print_fn)


def _evaluate(model, test_list, args, device, print_fn):
    auroc_sp_l, ap_sp_l, f1_sp_l = [], [], []
    auroc_px_l, ap_px_l, f1_px_l, aupro_px_l = [], [], [], []
    for item, test_data in zip(args.item_list, test_list):
        loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        results = evaluation_batch(model, loader, device, max_ratio=0.01, resize_mask=256)
        _print_metrics(item, results, print_fn)
        auroc_sp_l.append(results[0]); ap_sp_l.append(results[1]); f1_sp_l.append(results[2])
        auroc_px_l.append(results[3]); ap_px_l.append(results[4])
        f1_px_l.append(results[5]); aupro_px_l.append(results[6])

    print_fn("Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, "
             "P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}".format(
                 np.mean(auroc_sp_l), np.mean(ap_sp_l), np.mean(f1_sp_l),
                 np.mean(auroc_px_l), np.mean(ap_px_l), np.mean(f1_px_l),
                 np.mean(aupro_px_l)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INP-Former-MG train/test")

    parser.add_argument("--dataset", type=str, default="MVTec-AD",
                        choices=["MVTec-AD", "VisA", "Real-IAD"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./saved_results")
    parser.add_argument("--save-name", type=str, default="INP-Former-MG")

    # model
    parser.add_argument("--encoder", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--layer-groups", type=str, default="0,1,2,3,4,5,6,7",
                        help="semicolon-separated granularity groups, e.g. 0,1,2;3,4,5;6,7")
    parser.add_argument("--inp-nums", type=str, default="6",
                        help="comma-separated INP count per group, e.g. 4,4,4")
    parser.add_argument("--decoder-layers", type=str, default=None,
                        help="comma-separated decoder depth per group (default = group size)")

    # coherence / diversity / adaptive prototypes
    parser.add_argument("--coherence-mode", type=str, default="soft",
                        choices=["soft", "hard"])
    parser.add_argument("--coherence-temperature", type=float, default=0.1)
    parser.add_argument("--coherence-weight", type=float, default=0.2)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-prototypes", action="store_true")
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-sparsity-weight", type=float, default=0.0)

    # training
    parser.add_argument("--total-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"])

    args = parser.parse_args()
    args.save_name = (args.save_name +
                      f"_dataset={args.dataset}_Encoder={args.encoder}"
                      f"_groups={args.layer_groups.replace(';', 'x')}_inp={args.inp_nums}"
                      f"_coherence={args.coherence_mode}")

    # category info
    if args.dataset == "MVTec-AD":
        args.item_list = ["carpet", "grid", "leather", "tile", "wood", "bottle",
                          "cable", "capsule", "hazelnut", "metal_nut", "pill",
                          "screw", "toothbrush", "transistor", "zipper"]
    elif args.dataset == "VisA":
        args.item_list = ["candle", "capsules", "cashew", "chewinggum", "fryum",
                          "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4",
                          "pipe_fryum"]
    elif args.dataset == "Real-IAD":
        args.item_list = ["audiojack", "bottle_cap", "button_battery", "end_cap",
                          "eraser", "fire_hood", "mint", "mounts", "pcb",
                          "phone_battery", "plastic_nut", "plastic_plug",
                          "porcelain_doll", "regulator", "rolled_strip_base",
                          "sim_card_set", "switch", "tape", "terminalblock",
                          "toothbrush", "toy", "toy_brick", "transistor1", "usb",
                          "usb_adaptor", "u_block", "vcpill", "wooden_beads",
                          "woodstick", "zipper"]

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    main(args, print_fn)
