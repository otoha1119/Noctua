import torch
import torch.nn as nn

num_patch=16
class VitInputLayer(nn.Module):
    def __init__(self,
                 in_channels: int = 1,  # チャネル数
                 emb_dim: int = 384,    # 埋め込み後のベクトルの長さ
                 num_patch_row: int = num_patch,  # 高さ方向のパッチの数
                 image_size: int = 512):  # 入力画像の一辺の長さ
        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        self.num_patch=self.num_patch_row**2 #パッチの数
        self.patch_size=int(self.image_size//self.num_patch_row) #パッチの大きさ、一辺
        
        self.patch_emb_layer=nn.Conv2d( #パッチの分割と、埋め込みを行う層
            in_channels=self.in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.cls_token=nn.Parameter( #クラストークン
            torch.randn(1,1,emb_dim)
        )

        self.pos_emb=nn.Parameter( #位置埋め込み
            torch.randn(1,self.num_patch+1,emb_dim,requires_grad=False)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: # xは入力画像で(B,C,H,W)　バッチサイズ，チャンネル数，高さ，幅
         
        # z_0はViTへの入力で(B,N,D) バッチサイズ，トークン数，埋め込み後のベクトル長

        z_0=self.patch_emb_layer(x) # 位置埋め込み　(B,C,H,W)⇒(B,D,H/P,W/P) Pはパッチの一辺の大きさ

        z_0=z_0.flatten(2) #　(B,D,H/P,W/P)⇒(B,D,Np) Np=(H*W/P^2)
        
        z_0=z_0.transpose(1,2) #(B,D,Np)⇒(B,Np,D)

        z_0=torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0),1,1)),z_0],dim=1 #パッチの埋め込みの先頭にCLSトークンを付与，(B,Np,D)⇒(B,N,D) N=Np+1
        )                                                              #cls_token=(1,1,D)よりrepeatで(B,1,D)に変換
 
        z_0=z_0 + self.pos_emb

        return z_0 #(N,L,embdim)

