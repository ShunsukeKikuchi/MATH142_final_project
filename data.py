import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, lfilter
from sklearn.model_selection import StratifiedGroupKFold
import gc

class EEGDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        train: bool = True,
    ) -> None:
        self.all_data = df.with_columns(
                            pl.concat_list(['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']).alias("votes")
                        ).with_columns(
                            pl.col("votes").list.to_array(6),
                        pl.col("path").map_elements(eeg_from_parquet).alias('eeg')
                        ).select(['eeg', 'votes']).to_dicts()
        self.training = train

    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, idx: int):
        row = self.all_data[idx]
        y = np.array(row['votes']).astype('float32')
        X = np.array(row['eeg']).astype('float32') # shape (nsamples, raw_channels)
            
        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X) / 32.0  # scale down
        X = butter_lowpass_filter(
            X, cutoff=20, fs=200, order=6
        )

        return torch.tensor(X, dtype=torch.float32).permute(1,0),y

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff, fs=fs, btype="low")
    return lfilter(b, a, data)

def eeg_from_parquet(
    parquet_path: str,
) -> np.ndarray:
    eeg = pl.read_parquet(parquet_path).drop("EKG").cast(pl.Float32)

    # 2) centre-crop to CFG.nsamples rows (gracefully handles short files)
    rows = len(eeg)
    offset = max((rows - 10000) // 2, 0)
    eeg_slice = eeg[offset : offset + 10000]

    # 3) convert to NumPy and fill NaNs
    data = eeg_slice.to_numpy()
    col_mean = np.nanmean(data, axis=0)
    # columns that are entirely NaN → col_mean becomes NaN, so we replace later
    nan_rows, nan_cols = np.where(np.isnan(data))
    data[nan_rows, nan_cols] = col_mean[nan_cols]
    data = np.nan_to_num(data, nan=0.0)  # all-NaN columns → 0

    return data

def create_dfs(CFG):
    # Load training data
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    df = pl.read_csv(f"{CFG.dataset_path}/train.csv")

    for fold, (train_idx, valid_idx) in enumerate(sgkf.split(df, y=df["expert_consensus"], groups=df["patient_id"])):
        if fold == CFG.fold:
            break

    train_df = df[train_idx]
    valid_df = df[valid_idx]

    labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]

    train_labels = train_df.group_by('eeg_id', maintain_order=True).agg([
        *[pl.col(lbl+'_vote').sum() for lbl in labels],
        pl.len().alias('total_vote') 
    ]).with_columns(
        pl.sum_horizontal([pl.col(lbl+'_vote') for lbl in labels]).alias("total_vote").cast(pl.Float64)
    ).with_columns(
        *[pl.col(lbl+'_vote') / pl.col('total_vote') for lbl in labels],
        pl.concat_str(
        [
                pl.lit(f"{CFG.dataset_path}/train_eegs/"),
                pl.col('eeg_id').cast(pl.String),
                pl.lit(".parquet"),
            ],
        ).alias("path"),
    ).drop("total_vote")

    valid_labels = valid_df.group_by('eeg_id', maintain_order=True).agg([
        *[pl.col(lbl+'_vote').sum() for lbl in labels],
        pl.len().alias('total_vote') 
    ]).with_columns(
        pl.sum_horizontal([pl.col(lbl+'_vote') for lbl in labels]).alias("total_vote").cast(pl.Float64)
    ).with_columns(
        *[pl.col(lbl+'_vote') / pl.col('total_vote') for lbl in labels],
        pl.concat_str(
        [
                pl.lit(f"{CFG.dataset_path}/train_eegs/"),
                pl.col('eeg_id').cast(pl.String),
                pl.lit(".parquet"),
            ],
        ).alias("path"),
    ).drop("total_vote")


    del df, train_df, valid_df
    gc.collect()
    return train_labels, valid_labels