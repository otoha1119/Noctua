"""
MNISTのダウンロード関係のファイル
新規マシンではコンテナ内で以下のコマンドを実行
python -c "from MNISTdataset import MnistDataset; MnistDataset('/workspace/data', download=True)"
2回目位以降は不要.

追記) MNISTのデータセットは軽かったからリポジトリ直下に追加した
呼び出し時は以下のコマンド
from MNISTdataset import MnistDataset          # ← ファイル名に合わせて

root = "/workspace/MNIST"                      # ★ここだけ変更
train_ds = MnistDataset(root, train=True)

"""

import os
from pathlib import Path
from typing import Sequence, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

__all__ = [
    "MnistDataset",
    "mnist_loader",
    "mnist_anomaly_loader",
]

class MnistDataset(Dataset):
    """MNIST の中から指定した数字だけを含むサブセット Dataset。

    デフォルトでは手書き数字 **1 と 7** を抽出します。追加・変更したい場合は
    ``digits=(0, 2, 3)`` のように渡してください。
    """

    def __init__(
        self,
        root: str | os.PathLike,
        digits: Sequence[int] = (1, 7),
        train: bool = True,
        download: bool = False,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        if transform is None:
            transform = transforms.ToTensor()

        # フル MNIST をロード
        self._base = datasets.MNIST(
            root=str(Path(root).expanduser()),
            train=train,
            download=download,
            transform=transform,
        )

        # 対象ラベルのインデックスだけ残す
        targets = torch.as_tensor(self._base.targets)
        mask = torch.isin(targets, torch.tensor(list(digits)))
        self._indices = torch.where(mask)[0]

    # --------------------------------------------------------------
    # torch.utils.data.Dataset インターフェイス
    # --------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 (imperative mood)
        return self._indices.numel()

    def __getitem__(self, idx: int):
        base_idx = int(self._indices[idx])
        return self._base[base_idx]

# ------------------------------------------------------------------
# DataLoader 生成用のユーティリティ
# ------------------------------------------------------------------

def mnist_loader(
    root: str | os.PathLike,
    digits: Sequence[int] = (1, 7),
    train: bool = True,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = False,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:
    """指定ラベルのみを含む MNIST の DataLoader を返す便利関数。"""

    dataset = MnistDataset(
        root=root,
        digits=digits,
        train=train,
        download=download,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def mnist_anomaly_loader(
    root: str | os.PathLike,
    normal_digit: int = 1,
    abnormal_digit: int = 7,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = False,
    transform: Optional[transforms.Compose] = None,
) -> dict:
    """
    MNISTデータセットを正常データ（normal）と異常データ（abnormal）に分割し、
    train/val/test用のDataLoaderを返す。

    Parameters:
    -----------
    normal_digit : int
        正常データとして扱う数字（例: 1）。
    abnormal_digit : int
        異常データとして扱う数字（例: 7）。
    train_ratio : float
        訓練データの割合（例: 0.7）。
    val_ratio : float
        検証データの割合（例: 0.2）。
    その他の引数は `mnist_loader` と同様。

    Returns:
    --------
    loaders : dict
        train/val/test 用の DataLoader を含む辞書。
    """

    if transform is None:
        transform = transforms.ToTensor()

    # MNISTデータセットをロード
    full_dataset = datasets.MNIST(
        root=str(Path(root).expanduser()),
        train=True,
        download=download,
        transform=transform,
    )

    # 正常データ（normal）と異常データ（abnormal）を分離
    targets = torch.as_tensor(full_dataset.targets)
    normal_indices = torch.where(targets == normal_digit)[0]
    abnormal_indices = torch.where(targets == abnormal_digit)[0]

    # 異常データは一部だけ使用（例: 10%）
    abnormal_sample_size = int(len(abnormal_indices) * 0.1)
    abnormal_indices = abnormal_indices[:abnormal_sample_size]

    # 正常データと異常データを結合
    all_indices = torch.cat([normal_indices, abnormal_indices])
    all_targets = targets[all_indices]

    # データをシャッフル
    perm = torch.randperm(len(all_indices))
    all_indices = all_indices[perm]
    all_targets = all_targets[perm]

    # データを train/val/test に分割
    train_size = int(len(all_indices) * train_ratio)
    val_size = int(len(all_indices) * val_ratio)
    test_size = len(all_indices) - train_size - val_size

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    # DataLoader を作成
    def create_loader(indices):
        subset = torch.utils.data.Subset(full_dataset, indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return {
        "train": create_loader(train_indices),
        "val": create_loader(val_indices),
        "test": create_loader(test_indices),
    }

# ------------------------------------------------------------------
# スモークテスト
# ------------------------------------------------------------------

if __name__ == "__main__":
    loader = mnist_loader("~/data/mnist", download=False, batch_size=256)
    print(f"Loaded {len(loader.dataset)} samples (digits 1 & 7).")
    imgs, labels = next(iter(loader))
    print("Batch shape:", imgs.shape, "Unique labels:", labels.unique().tolist())
