import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from loguru import logger
from .model import PriceTransformer

class TransformerTrainer:
    def __init__(self,n_features,seq_len=32,epochs=8,batch_size=32,lr=2e-3,device=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')); self.seq_len=seq_len; self.epochs=epochs; self.batch_size=batch_size
        self.model=PriceTransformer(n_features=n_features,seq_len=seq_len).to(self.device); self.mean=None; self.std=None
    def fit(self,X,y):
        self.mean=np.nanmean(X,axis=(0,1),keepdims=True); self.std=np.nanstd(X,axis=(0,1),keepdims=True)+1e-6; X=(np.nan_to_num(X)-self.mean)/self.std
        dl=DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(np.nan_to_num(y),dtype=torch.float32)),batch_size=self.batch_size,shuffle=False)
        opt=torch.optim.AdamW(self.model.parameters(),lr=2e-3,weight_decay=1e-4); loss_fn=torch.nn.HuberLoss(); amp=self.device.type=='cuda'; scaler=torch.amp.GradScaler('cuda',enabled=amp)
        logger.info(f'Transformer device={self.device}, AMP={amp}, samples={len(dl.dataset):,}')
        self.model.train()
        for ep in range(1,self.epochs+1):
            ls=[]
            for xb,yb in dl:
                xb,yb=xb.to(self.device),yb.to(self.device); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp): loss=loss_fn(self.model(xb),yb)
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0); scaler.step(opt); scaler.update(); ls.append(float(loss.detach().cpu()))
            logger.info(f'Epoch {ep}/{self.epochs} loss={np.mean(ls):.6f}')
        return self
    @torch.no_grad()
    def predict(self,X):
        X=(np.nan_to_num(X)-self.mean)/self.std; self.model.eval(); out=[]
        for i in range(0,len(X),self.batch_size): out.append(self.model(torch.tensor(X[i:i+self.batch_size],dtype=torch.float32,device=self.device)).float().cpu().numpy())
        return np.concatenate(out)
