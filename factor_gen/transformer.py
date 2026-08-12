from __future__ import annotations
import numpy as np

class TransformerFactor:
    """Small 4GB-GPU-safe Transformer; falls back to a deterministic score if torch is absent."""
    def __init__(self, seq_len=32, d_model=32, heads=4, layers=2, epochs=8, batch_size=32):
        self.seq_len,self.d_model,self.heads,self.layers,self.epochs,self.batch_size=seq_len,d_model,heads,layers,epochs,batch_size
        self.model=None

    def fit_predict(self, X, y):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return X[:,4] - 0.5*X[:,6]
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n=len(X); X=np.nan_to_num(X).astype("float32")
        mu=X.mean(0,keepdims=True); sd=X.std(0,keepdims=True)+1e-6; X=(X-mu)/sd
        # Point-wise token projection; rolling windows are formed without shuffling time.
        proj=nn.Linear(X.shape[1],self.d_model)
        enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(self.d_model,self.heads,batch_first=True),self.layers)
        head=nn.Linear(self.d_model,1)
        net=nn.Sequential(proj,enc,head).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
        loss_fn=nn.HuberLoss()
        Xt=torch.tensor(X,device=device); yt=torch.tensor(y,device=device).float()
        for epoch in range(self.epochs):
            net.train(); total=0.0; steps=0
            for end in range(self.seq_len,n,self.batch_size):
                starts=max(self.seq_len,end-self.batch_size+1)
                windows=torch.stack([Xt[i-self.seq_len:i] for i in range(starts,end)])
                target=yt[starts:end]
                pred=net(windows).squeeze(-1)[:len(target)]
                loss=loss_fn(pred,target); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
                total+=float(loss); steps+=1
            print(f"Transformer epoch {epoch+1}/{self.epochs} loss={total/max(steps,1):.6f} device={device}")
        net.eval(); out=np.full(n,np.nan,dtype="float32")
        with torch.no_grad():
            for end in range(self.seq_len,n,self.batch_size):
                starts=max(self.seq_len,end-self.batch_size+1)
                windows=torch.stack([Xt[i-self.seq_len:i] for i in range(starts,end)])
                p=net(windows).squeeze(-1).detach().cpu().numpy()
                out[starts:end]=p[:end-starts]
        self.model=net
        return out
