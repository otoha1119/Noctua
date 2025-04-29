"""
MNISTのダウンロード関係のファイル
新規マシンではコンテナ内で以下のコマンドを実行

python -c "from MNISTdataset import MnistDataset; MnistDataset('/workspace/data', download=True)"

2回目位以降は不要.

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

# ------------------------------------------------------------------
# スモークテスト
# ------------------------------------------------------------------

if __name__ == "__main__":
    loader = mnist_loader("~/data/mnist", download=False, batch_size=256)
    print(f"Loaded {len(loader.dataset)} samples (digits 1 & 7).")
    imgs, labels = next(iter(loader))
    print("Batch shape:", imgs.shape, "Unique labels:", labels.unique().tolist())
