''''
import torch
import torch.nn as nn
import numpy as np

numpatch = 16
embdim = 384

class Decoder(nn.Module):
    def __init__(self, emb_dim: int = embdim, image_size: int = 512, num_patch_row: int = numpatch):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.image_size = image_size
        self.num_patch_row = num_patch_row
        self.patch_size = int(image_size // num_patch_row)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 転置畳み込み
            nn.InstanceNorm2d(256),  # インスタンス正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 転置畳み込み
            nn.InstanceNorm2d(128),  # インスタンス正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 転置畳み込み
            nn.InstanceNorm2d(64),  # インスタンス正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 転置畳み込み
            nn.InstanceNorm2d(32),  # インスタンス正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 転置畳み込み
            nn.InstanceNorm2d(16),  # インスタンス正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # 転置畳み込み
            nn.InstanceNorm2d(1),  # インスタンス正規化
            nn.ReLU(True),  # ReLU
            nn.Tanh()  # Tanh
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, emb_dim = x.shape
        assert emb_dim == self.emb_dim, f'Expected embedding dimension {self.emb_dim}, but got {emb_dim}'
        patch_size_sqrt = int(np.sqrt(num_patches - 1))
        assert patch_size_sqrt == self.num_patch_row, f'Expected number of patches per row {self.num_patch_row}, but got {patch_size_sqrt}'
        x = x[:, 1:, :]
        x = x.permute(0, 2, 1).contiguous().view(batch_size, self.emb_dim, self.num_patch_row, self.num_patch_row)  # 転置してreshape
        reconstructed_img = self.decoder(x)  # デコーダーを通す
        return reconstructed_img



'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
numpatch=16
embdim=512
class Decoder(nn.Module):
    def __init__(self, emb_dim: int = embdim, image_size: int = 512, num_patch_row: int = numpatch):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.image_size = image_size
        self.num_patch_row = num_patch_row
        self.patch_size = int(image_size // num_patch_row)

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(emb_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 転置畳み込み
            # nn.InstanceNorm2d(512),  # バッチ正規化
            # nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(emb_dim, 256, kernel_size=4, stride=2, padding=1, output_padding=0),  # 転置畳み込み
            nn.InstanceNorm2d(256),  # バッチ正規化
            nn.LeakyReLU(True),  # ReLU
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0,),  # 転置畳み込み
            nn.InstanceNorm2d(128),  # バッチ正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),  # 転置畳み込み
            nn.InstanceNorm2d(64),  # バッチ正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # 転置畳み込み
            nn.InstanceNorm2d(32),  # バッチ正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=0),  # 転置畳み込み
            nn.InstanceNorm2d(16),  # バッチ正規化
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),  # 転置畳み込み
            nn.InstanceNorm2d(8),
            nn.ReLU(True),  # ReLU
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),  # 転置畳み込み
            # nn.InstanceNorm2d(1),
            nn.ReLU(True),
            # nn.UpsamplingBilinear2d((128, 128)),  # バイリニアアップサンプリング
            nn.Tanh()  # Tanh
            #nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, emb_dim = x.shape
        assert emb_dim == self.emb_dim, f'Expected embedding dimension {self.emb_dim}, but got {emb_dim}'
        patch_size_sqrt = int(np.sqrt(num_patches - 1))
        assert patch_size_sqrt == self.num_patch_row, f'Expected number of patches per row {self.num_patch_row}, but got {patch_size_sqrt}'
        x = x[:, 1:, :]
        x = x.permute(0, 2, 1).contiguous().view(batch_size, self.emb_dim, self.num_patch_row, self.num_patch_row)  # 転置してreshape
        reconstructed_img = self.decoder(x)  # デコーダーを通す
        return reconstructed_img
    

# モデルのインスタンス化
# decoder = Decoder(emb_dim=512, image_size=128, num_patch_row=8)

# # ランダムな入力データ (B, Np+1, D_emb) 
# batch_size = 1
# num_patches = 8 * 8 + 1  # クラストークンを含むパッチの数
# x = torch.randn(batch_size, num_patches, 512)

# # デコーダーを通して出力を生成
# output = decoder(x)

# print(output.shape)

