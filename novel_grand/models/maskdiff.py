from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from novel_grand.models.training import _safe_torch_load, _to_numpy



def _standardize_fit_tokens(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)



def _standardize_apply_tokens(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)



def _standardize_fit_global(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)



def _standardize_apply_global(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)



class MaskDiffDenoiser(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        global_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        max_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, d_model)
        self.global_proj = nn.Linear(global_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, token_x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        b, k, _ = token_x.shape
        h = self.token_proj(token_x)
        h = h + self.pos_emb[:, :k, :]
        g = self.global_proj(global_x).unsqueeze(1)
        h = h + g
        h = self.encoder(h)
        return self.out(h).squeeze(-1)


@dataclass
class MaskDiffModel:
    model: nn.Module
    token_mean: np.ndarray
    token_std: np.ndarray
    global_mean: np.ndarray
    global_std: np.ndarray
    device: str

    def predict_logits(self, token_x: np.ndarray, global_x: np.ndarray) -> np.ndarray:
        tx = _standardize_apply_tokens(np.asarray(token_x, dtype=np.float32), self.token_mean, self.token_std)
        gx = _standardize_apply_global(np.asarray(global_x, dtype=np.float32), self.global_mean, self.global_std)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(tx).to(self.device),
                torch.from_numpy(gx).to(self.device),
            ).cpu().numpy()
        return logits.astype(np.float32)

    def predict_prob(self, token_x: np.ndarray, global_x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(token_x, global_x)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)



def fit_maskdiff(
    token_x: np.ndarray,
    global_x: np.ndarray,
    y: np.ndarray,
    loss_mask: np.ndarray,
    cfg,
    *,
    device: str = "cpu",
):
    if token_x.ndim != 3:
        raise ValueError(f"token_x must be [N,K,F], got {token_x.shape}")
    if global_x.ndim != 2:
        raise ValueError(f"global_x must be [N,G], got {global_x.shape}")
    if y.shape[:2] != token_x.shape[:2]:
        raise ValueError(f"target shape mismatch: token_x={token_x.shape}, y={y.shape}")
    if loss_mask.shape != y.shape:
        raise ValueError(f"loss_mask mismatch: expected {y.shape}, got {loss_mask.shape}")

    token_x = np.asarray(token_x, dtype=np.float32)
    global_x = np.asarray(global_x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    loss_mask = np.asarray(loss_mask, dtype=np.float32)

    tok_mean, tok_std = _standardize_fit_tokens(token_x)
    glb_mean, glb_std = _standardize_fit_global(global_x)
    token_std_x = _standardize_apply_tokens(token_x, tok_mean, tok_std)
    global_std_x = _standardize_apply_global(global_x, glb_mean, glb_std)

    ds = TensorDataset(
        torch.from_numpy(token_std_x),
        torch.from_numpy(global_std_x),
        torch.from_numpy(y),
        torch.from_numpy(loss_mask),
    )
    n_total = len(ds)
    n_val = max(1, int(round(float(cfg["training"].get("validation_fraction", 0.1)) * n_total))) if n_total > 1 else 1
    n_train = max(n_total - n_val, 1)
    if n_train + n_val > n_total:
        n_val = n_total - n_train
    tr_ds, va_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(1234))

    tr_cfg = cfg["training"]
    d_model = int(tr_cfg.get("maskdiff_d_model", 48))
    nhead = int(tr_cfg.get("maskdiff_num_heads", 4))
    num_layers = int(tr_cfg.get("maskdiff_num_layers", 2))
    ff_dim = int(tr_cfg.get("maskdiff_ffn_dim", 96))

    model = MaskDiffDenoiser(
        token_dim=int(token_x.shape[-1]),
        global_dim=int(global_x.shape[-1]),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        max_len=int(token_x.shape[1]),
        dropout=float(tr_cfg.get("maskdiff_dropout", 0.0)),
    ).to(device)

    pos = float((y * loss_mask).sum())
    neg = float(((1.0 - y) * loss_mask).sum())
    pos_weight = neg / max(pos, 1.0)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg["weight_decay"]),
    )

    tr_loader = DataLoader(tr_ds, batch_size=min(int(tr_cfg.get("maskdiff_batch_size", 256)), max(n_train, 1)), shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=min(int(tr_cfg.get("maskdiff_batch_size", 256)), max(n_val, 1)), shuffle=False)

    best_state = None
    best_val = float("inf")

    for _ in range(int(tr_cfg["epochs"])):
        model.train()
        for txb, gxb, yb, mb in tr_loader:
            txb = txb.to(device)
            gxb = gxb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            logits = model(txb, gxb)
            loss = bce(logits, yb)
            loss = (loss * mb).sum() / torch.clamp(mb.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        n_seen = 0
        val_loss = 0.0
        with torch.no_grad():
            for txb, gxb, yb, mb in va_loader:
                txb = txb.to(device)
                gxb = gxb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                logits = model(txb, gxb)
                loss = bce(logits, yb)
                denom = torch.clamp(mb.sum(), min=1.0)
                loss = (loss * mb).sum() / denom
                bs = txb.shape[0]
                val_loss += float(loss.item()) * bs
                n_seen += bs
        val_loss /= max(n_seen, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    meta = {
        "best_val_loss": float(best_val),
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "token_dim": int(token_x.shape[-1]),
        "global_dim": int(global_x.shape[-1]),
        "max_len": int(token_x.shape[1]),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "ff_dim": ff_dim,
    }
    return model, tok_mean, tok_std, glb_mean, glb_std, meta



def save_maskdiff_bundle(
    path: str | Path,
    model: nn.Module,
    token_mean: np.ndarray,
    token_std: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    meta: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "token_mean": torch.from_numpy(_to_numpy(token_mean)),
            "token_std": torch.from_numpy(_to_numpy(token_std)),
            "global_mean": torch.from_numpy(_to_numpy(global_mean)),
            "global_std": torch.from_numpy(_to_numpy(global_std)),
            "meta": dict(meta),
        },
        path,
    )



def load_maskdiff(path: str | Path, *, device: str = "cpu") -> MaskDiffModel:
    ckpt = _safe_torch_load(path, device=device)
    meta = dict(ckpt["meta"])
    model = MaskDiffDenoiser(
        token_dim=int(meta["token_dim"]),
        global_dim=int(meta["global_dim"]),
        d_model=int(meta["d_model"]),
        nhead=int(meta["nhead"]),
        num_layers=int(meta["num_layers"]),
        ff_dim=int(meta["ff_dim"]),
        max_len=int(meta["max_len"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return MaskDiffModel(
        model=model,
        token_mean=_to_numpy(ckpt["token_mean"]),
        token_std=_to_numpy(ckpt["token_std"]),
        global_mean=_to_numpy(ckpt["global_mean"]),
        global_std=_to_numpy(ckpt["global_std"]),
        device=device,
    )
