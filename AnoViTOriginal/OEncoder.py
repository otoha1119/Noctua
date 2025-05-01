import torch
import torch.nn as nn
from MHSA import MultiHeadSelfAttention
from ViTInputlayer import VitInputLayer

numpatch=16
embdim=384
class VitEncoderBlock(nn.Module):
    def __init__(self, 
                 emb_dim: int = embdim, 
                 head: int = 8, 
                 hidden_dim: int = embdim * 4, 
                 dropout: float = 0):
        
        super(VitEncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(emb_dim)  # LayerNorm
        self.msa = MultiHeadSelfAttention(emb_dim=emb_dim, 
                                          head=head, 
                                          dropout=dropout)  # MSA
        self.ln2 = nn.LayerNorm(emb_dim)  # LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),  # MLP
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.msa(self.ln1(z)) + z  # MSAと残差接続
        out = self.mlp(self.ln2(out)) + out  # MLPと残差接続
        return out #(N,L,embdim) 

# ViTエンコーダー
class VitEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int = 1, 
                 emb_dim: int = embdim, 
                 num_patch_row: int = numpatch, 
                 image_size: int = 512,
                 num_blocks: int = 8, 
                 head: int = 8, 
                 hidden_dim: int =embdim * 4, 
                 dropout: float = 0.):
        super(VitEncoder, self).__init__()
        self.input_layer = VitInputLayer(in_channels, 
                                         emb_dim, 
                                         num_patch_row, 
                                         image_size)  # 入力レイヤー
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(emb_dim=emb_dim, 
                            head=head, 
                            hidden_dim=hidden_dim, 
                            dropout=dropout)
            for _ in range(num_blocks)
        ])  # エンコーダーブロックのスタック

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)  # 入力レイヤーを通す
        features = self.encoder(out)  # エンコーダーブロックを通す
        return features