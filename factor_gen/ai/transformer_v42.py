from __future__ import annotations
import numpy as np


class V42Transformer:
    def __init__(self, seq_len=32, epochs=5, batch_size=32, d_model=32, heads=4, layers=2, infer_batch_size=128, learning_rate=2e-3):
        self.seq_len, self.epochs, self.batch_size = seq_len, epochs, batch_size
        self.d_model, self.heads, self.layers = d_model, heads, layers
        self.infer_batch_size = infer_batch_size
        self.learning_rate = learning_rate
        self.net = None; self.mu = None; self.sd = None; self.device = None

    def _windows(self, X, y=None):
        n=max(0,len(X)-self.seq_len+1)
        if n==0: return np.empty((0,self.seq_len,X.shape[1]),np.float32), np.empty((0,),np.float32) if y is not None else None
        wx=np.lib.stride_tricks.sliding_window_view(X,(self.seq_len,X.shape[1]))[:,0,:,:].copy().astype(np.float32,copy=False)[:n]
        wy=y[self.seq_len-1:self.seq_len-1+n].astype(np.float32,copy=False) if y is not None else None
        if y is not None and len(wx)!=len(wy): raise RuntimeError(f"Transformer window/target mismatch: X={len(wx)} y={len(wy)}")
        return wx,wy

    def _windows_by_symbol(self, frame, features, target):
        xs=[]; ys=[]
        ordered=frame.sort_values(["symbol","eob"],kind="stable")
        for _,part in ordered.groupby("symbol",sort=False):
            X=part[features].to_numpy(np.float32); y=part[target].to_numpy(np.float32)
            if len(X)<self.seq_len: continue
            wx,wy=self._windows(X,y)
            if len(wx): xs.append(wx); ys.append(wy)
        if not xs: return np.empty((0,self.seq_len,len(features)),np.float32),np.empty((0,),np.float32)
        X_all=np.concatenate(xs,axis=0); y_all=np.concatenate(ys,axis=0)
        if len(X_all)!=len(y_all): raise RuntimeError(f"Transformer dataset mismatch: X={len(X_all)} y={len(y_all)}")
        return X_all,y_all

    def _get_device(self,torch):
        if torch.cuda.is_available():
            device=torch.device("cuda"); print(f"[Transformer] CUDA enabled: {torch.cuda.get_device_name(0)}"); print(f"[Transformer] CUDA memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB"); return device
        print("[Transformer] WARNING: CUDA unavailable; using CPU"); print(f"[Transformer] torch={torch.__version__} torch.version.cuda={torch.version.cuda}"); return torch.device("cpu")

    def _iter_predict_batches(self,X):
        total=max(0,len(X)-self.seq_len+1)
        for first in range(0,total,self.infer_batch_size):
            last=min(first+self.infer_batch_size,total); wx=np.empty((last-first,self.seq_len,X.shape[1]),dtype=np.float32)
            for j,i in enumerate(range(first,last)): wx[j]=X[i:i+self.seq_len]
            yield first+self.seq_len-1,wx

    def fit(self,frame,features,target):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.net=None; print("[Transformer] PyTorch not installed; AI score disabled"); return self
        ordered=frame.sort_values(["symbol","eob"],kind="stable").copy()
        X_all=ordered[features].to_numpy(np.float32); self.mu=np.nanmean(X_all,axis=0); self.sd=np.nanstd(X_all,axis=0)+1e-6
        normalized=ordered.copy(); normalized.loc[:,features]=np.nan_to_num((X_all-self.mu)/self.sd)
        wx,wy=self._windows_by_symbol(normalized,features,target)
        if len(wx)==0: print("[Transformer] no training windows"); return self
        wy=np.nan_to_num(wy.astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
        self.device=self._get_device(torch); proj=nn.Linear(wx.shape[2],self.d_model); enc_layer=nn.TransformerEncoderLayer(d_model=self.d_model,nhead=self.heads,batch_first=True); enc=nn.TransformerEncoder(enc_layer,self.layers); head=nn.Linear(self.d_model,1); self.net=nn.ModuleList([proj,enc,head]).to(self.device)
        opt=torch.optim.AdamW(self.net.parameters(),lr=self.learning_rate,weight_decay=1e-4); loss_fn=nn.HuberLoss(); tx_cpu=torch.from_numpy(wx); ty_cpu=torch.from_numpy(wy); n=len(tx_cpu)
        if n!=len(ty_cpu): raise RuntimeError(f"Transformer training data mismatch: X={n} y={len(ty_cpu)}")
        print(f"[Transformer] training samples={n} batch={self.batch_size} lr={self.learning_rate} device={self.device} chunked=True")
        for ep in range(self.epochs):
            self.net.train(); total_loss=0.0; batches=0; order_idx=torch.randperm(n,device="cpu")
            for start in range(0,n,self.batch_size):
                idx=order_idx[start:start+self.batch_size]; xb=tx_cpu[idx].to(self.device,non_blocking=True); yb=ty_cpu[idx].to(self.device,non_blocking=True); h=self.net[0](xb); h=self.net[1](h); pred=self.net[2](h[:,-1,:]).squeeze(-1); loss=loss_fn(pred,yb); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0); opt.step(); total_loss+=loss.detach().item(); batches+=1; del xb,yb,h,pred,loss
            print(f"[Transformer] epoch {ep+1}/{self.epochs} loss={total_loss/max(1,batches):.6f} device={self.device}")
        return self

    def _predict_one(self,frame,features,out,index_positions=None):
        import torch
        ordered=frame.sort_values(["symbol","eob"],kind="stable").copy() if "symbol" in frame.columns else frame.copy()
        X=ordered[features].to_numpy(np.float32); X=np.nan_to_num((X-self.mu)/self.sd); total=max(0,len(X)-self.seq_len+1)
        if total==0: return
        device=next(self.net.parameters()).device; self.net.eval(); done=0
        local_positions=np.asarray(ordered.index,dtype=np.int64)
        with torch.inference_mode():
            for start_pos,wx in self._iter_predict_batches(X):
                tx=torch.from_numpy(wx).to(device,non_blocking=True); h=self.net[0](tx); h=self.net[1](h); pred=self.net[2](h[:,-1,:]).squeeze(-1).cpu().numpy(); end_pos=start_pos+len(pred); target_positions=local_positions[start_pos:end_pos]
                if index_positions is None: out[target_positions]=pred
                else: out[index_positions[target_positions]]=pred
                done+=len(pred); del tx,h
        print(f"[Transformer] inference {done}/{total} samples batch={self.infer_batch_size} device={device}")

    def predict(self,frame,features):
        out=np.full(len(frame),np.nan,np.float32)
        if self.net is None or len(frame)==0: return out
        if "symbol" in frame.columns:
            positions=frame.groupby("symbol",sort=False).indices
            for _,pos in positions.items():
                pos=np.asarray(pos,dtype=np.int64); sub=frame.iloc[pos].copy(); self._predict_one(sub,features,out,pos)
        else: self._predict_one(frame,features,out)
        return out
