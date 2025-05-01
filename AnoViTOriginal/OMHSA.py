import torch
import torch.nn as nn
import torch.nn.functional as F

embdim=384

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self,
                 emb_dim:int=embdim, #埋め込み後のベクトル長
                 head:int=8, #ヘッドの数
                 dropout:float=0. #ドロップアウト率
                 ):
        
        super(MultiHeadSelfAttention,self).__init__()
        self.head=head
        self.emb_dim=emb_dim
        self.head_dim=emb_dim//head
        self.sqrt_dh=self.head_dim**0.5 #D_hの2乗根　qk^Tを割る係数
        
        #入力をq,k,vに埋め込む線形層
        self.w_q=nn.Linear(emb_dim,emb_dim,bias=False)
        self.w_k=nn.Linear(emb_dim,emb_dim,bias=False)
        self.w_v=nn.Linear(emb_dim,emb_dim,bias=False)

        self.attn_drop=nn.Dropout(dropout)

        #MHSAの結果を出力に埋め込むための線形層
        self.w_o=nn.Sequential(
            nn.Linear(emb_dim,emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor: #　zはMHSAの入力　(B,N,D) Dはベクトルの長さ　返り値out (B,N,D) バッチサイズ，トークン数，埋め込み後のベクトル長

        batch_size,num_patch,_=z.size()
        
        #埋め込み
        q=self.w_q(z)
        k=self.w_k(z)
        v=self.w_v(z)
        
        #q,k,vをヘッドに分ける⇒　ベクトルをヘッドの個数に分け、self attentionできるように(B,h,N,D//h)の形に変更する　(バッチサイズ，ヘッド，トークン数，パッチのベクトル)
        q=q.view(batch_size,num_patch,self.head,self.head_dim)
        k=k.view(batch_size,num_patch,self.head,self.head_dim)
        v=v.view(batch_size,num_patch,self.head,self.head_dim)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)
        
        #内積
        k_T=k.transpose(2,3)

        dots=(q @ k_T)/self.sqrt_dh
        #列方向にソフトマックス
        attn=F.softmax(dots,dim=-1)

        attn=self.attn_drop(attn)
        
        #加重和
        out=attn @ v
        out=out.transpose(1,2)
        out=out.reshape(batch_size,num_patch,self.emb_dim)

        out=self.w_o(out)
        return out
