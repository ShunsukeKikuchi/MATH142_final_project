import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import gc

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from model3 import resnext
from data import EEGDataset, create_dfs

class CFG:
    # basic
    model_name = "eegnet_v2"
    seed = 42
    fold = 0

    # training setting
    n_epoch = 60*60
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=32
    lr = 1e-3
    autocast=True # used for training, not for validation

    dataset_path = "."

def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    config = CFG()
    set_seed(config.seed)

    train_labels, valid_labels = create_dfs(config)
    train_dataset = EEGDataset(train_labels, train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=16)
    valid_dataset = EEGDataset(valid_labels, train=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=16)

    model = resnext(num_classes=6, in_channels=19, width_mult = 1.0)
    model.to("cuda")
    print("number of parameters: ", sum(p.numel() for p in model.parameters()))
    model = torch.compile(model)

    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    best_loss = np.inf

    for epoch in range(1, config.n_epoch+1):
        avg_loss = train(model, train_loader, optimizer, kl_loss, epoch)
        avg_val_loss = valid(model, valid_loader, kl_loss, epoch)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model3.pth")
            print(f">>> New best model saved (KL={best_loss:.4f}) <<<")

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for spec, target in train_loader:
        spec   = spec.to(CFG.device)
        target = target.to(CFG.device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=CFG.autocast):
            log_pred = model(spec).log_softmax(dim=1)
            target_clipped = torch.clamp(target, 1e-8, 1.0)
            loss = criterion(log_pred, target_clipped)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"*** Epoch {epoch} TRAINING COMPLETE. Avg Loss: {avg_loss:.4f} ***")
    torch.cuda.empty_cache(); gc.collect()
    return avg_loss


def valid(model, valid_loader, criterion, epoch):
    model.eval()
    all_log_pred, all_target = [], []

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for spec, target in valid_loader:
            spec   = spec.to(CFG.device)
            target = target.to(CFG.device)

            log_pred = model(spec).log_softmax(dim=1)
            all_log_pred.append(log_pred)
            all_target.append(target)

        log_pred = torch.cat(all_log_pred, dim=0)
        target   = torch.cat(all_target , dim=0)
        target   = torch.clamp(target, 1e-8, 1.0)

        val_loss = criterion(log_pred, target)

    print(f"=== Epoch {epoch} VALIDATION: KL Divergence = {val_loss:.4f} ===")
    torch.cuda.empty_cache(); gc.collect()
    return val_loss

if __name__ == "__main__":
    main()
