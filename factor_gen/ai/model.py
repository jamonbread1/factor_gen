import torch
from torch import nn

class PriceTransformer(nn.Module):
    def __init__(self,n_features=10,seq_len=32,hidden_dim=32,heads=4,layers=2,dropout=0.1):
        super().__init__(); self.proj=nn.Linear(n_features,hidden_dim); self.pos=nn.Parameter(torch.zeros(1,seq_len,hidden_dim))
        enc=nn.TransformerEncoderLayer(hidden_dim,heads,hidden_dim*2,dropout,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,layers); self.head=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,1))
    def forward(self,x): return self.head(self.encoder(self.proj(x)+self.pos[:,:x.size(1)] )[:,-1]).squeeze(-1)
