from __future__ import annotations
import numpy as np

class V42Transformer:
    def __init__(self, seq_len=32, epochs=5, batch_size=32, d_model=32, heads=4, layers=2):
        self.seq_len,self.epochs,self.batch_size=seq_len,epochs,batch_size
        self.d_model,self.heads,self.layers=d_model,heads,layers
        self.net=None; self.mu=None; self.sd=None

    def _windows(self, X, y=None):
        xs=[]; ys=[]
        for i in range(self.seq_len,len(X)):
            xs.append(X[i-self.seq_len:i]);
            if y is not None: ys.append(y[i])
        return np.asarray(xs,np.float32), np.asarray(ys,np.float32) if y is not None else None

    def fit(self, frame, features, target):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.net=None; return self
        X=frame[features].to_numpy(np.float32); y=frame[target].to_numpy(np.float32)
        self.mu=np.nanmean(X,axis=0); self.sd=np.nanstd(X,axis=0)+1e-6; X=np.nan_to_num((X-self.mu)/self.sd)
        wx,wy=self._windows(X,y)
        if len(wx)==0: return self
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proj=nn.Linear(X.shape[1],self.d_model)
        enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(self.d_model,self.heads,batch_first=True),self.layers)
        head=nn.Linear(self.d_model,1)
        self.net=nn.Sequential(proj,enc,head).to(device)
        opt=torch.optim.AdamW(self.net.parameters(),lr=2e-3,weight_decay=1e-4)
        loss_fn=nn.HuberLoss()
        tx=torch.tensor(wx,device=device); ty=torch.tensor(wy,device=device)
        for ep in range(self.epochs):
            self.net.train(); total=0.0
            order=torch.arange(len(tx),device=device)
            for start in range(0,len(order),self.batch_size):
                idx=order[start:start+self.batch_size]; pred=self.net(tx[idx]).squeeze(-1)
                loss=loss_fn(pred,ty[idx]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0); opt.step(); total+=float(loss)
            print(f"[Transformer] epoch {ep+1}/{self.epochs} loss={total/max(1,(len(order)+self.batch_size-1)//self.batch_size):.6f} device={device}")
        return self

    def predict(self, frame, features):
        out=np.full(len(frame),np.nan,np.float32)
        if self.net is None: return out
        import torch
        X=frame[features].to_numpy(np.float32); X=np.nan_to_num((X-self.mu)/self.sd)
        wx,_=self._windows(X)
        if len(wx)==0:return out
        device=next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            pred=self.net(torch.tensor(wx,device=device)).squeeze(-1).cpu().numpy()
        out[self.seq_len:]=pred
        return out
