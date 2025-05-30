import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import gc
import timm

import numpy as np
import polars as pl

from tqdm import tqdm

from torchaudio import transforms as T
from torchvision.transforms import v2

from src.data import create_dfs

class CFG:
    # basic
    model_name = "resnet50"
    seed = 42
    fold = 0

    # training setting
    n_epoch = 15*10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=128
    lr = 1e-3
    img_size = (257, 600)
    train_transform=v2.Resize(img_size)
    valid_transform=v2.Resize(img_size)
    autocast=True # used for training, not for validation

    dataset_path = "/u/home/s/skikuchi/scratch/Kaggle/hms/data/"

def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def eeg2spec(eeg):
    # x: (C,T) or (B, C, T)
    input_len = eeg.shape[-1]
    transform = T.Spectrogram(n_fft = 512,
                          win_length = 64,
                          hop_length = input_len // 600, # 
                          power = 1)
    spec = transform(eeg)**0.8
    spec = torch.nan_to_num(spec)
    spec = F.normalize(spec)
    return spec

def eeg2spec2(eeg):
    # x: (C,T) or (B, C, T)
    eeg = pl.read_parquet(eeg)
    eeg = eeg.drop("EKG").to_torch().transpose(1, 0)
    input_len = eeg.shape[-1]
    transform = T.Spectrogram(n_fft = 512,
                          win_length = 64,
                          hop_length = input_len // 600, # 
                          power = 1)
    spec = transform(eeg)**0.8
    spec = torch.nan_to_num(spec)
    spec = F.normalize(spec)
    return spec

class HmsDataset(Dataset):
    def __init__(self, df: pl.DataFrame, transform=None):
        '''
        in train/valid - set train True
        in inference - set train False
        '''
        self.all_data = df.with_columns(
                            pl.concat_list(['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']).alias("votes")
                        ).with_columns(
                            pl.col("votes").list.to_array(6),
                        pl.col("path").map_elements(eeg2spec2).alias('spec')
                        ).select(['spec', 'votes']).to_dicts()
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx: int):
        X = self.all_data[idx]['spec']
        y = torch.tensor(self.all_data[idx]['votes'])
        if self.transform is not None:
            X = self.transform(X)
        return X, y

def main():
    config = CFG()
    set_seed(config.seed)

    train_labels, valid_labels = create_dfs(config)
    train_dataset = HmsDataset(train_labels, transform=CFG.train_transform)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=16)
    valid_dataset = HmsDataset(valid_labels, transform=CFG.valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=16)

    model = timm.create_model(CFG.model_name, pretrained=True, num_classes=6, in_chans=19)
    model.to("cuda")
    print("number of parameters: ", sum(p.numel() for p in model.parameters()))
    model = torch.compile(model)

    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    best_loss = np.inf

    for epoch in range(1, CFG.n_epoch+1):
        avg_loss = train(model, train_loader, optimizer, kl_loss, epoch)
        avg_val_loss = valid(model, valid_loader, kl_loss, epoch)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"weights/{CFG.model_name}_best_model.pth")
            print(f">>> New best model saved (KL={best_loss:.4f}) <<<")

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for spec, target in tqdm(train_loader):
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
        for spec, target in tqdm(valid_loader):
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