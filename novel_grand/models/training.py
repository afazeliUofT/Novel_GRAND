from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from novel_grand.models.mlp import TabularMLP


def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)



def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)



def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)



def _safe_torch_load(path: str | Path, device: str = "cpu"):
    """Load local checkpoints robustly across PyTorch weights_only defaults.

    PyTorch 2.6+ defaults torch.load(..., weights_only=True), which rejects
    checkpoint bundles that include NumPy arrays. These model bundles are
    trusted local artifacts produced by this repo, so falling back to
    weights_only=False is acceptable.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch without weights_only support.
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


@dataclass
class SnapshotSelectorModel:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    device: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        xx = _standardize_apply(x.astype(np.float32), self.mean, self.std)
        with torch.no_grad():
            out = self.model(torch.from_numpy(xx).to(self.device)).reshape(-1).cpu().numpy()
        return out.astype(np.float32)

    def select(self, x: np.ndarray) -> int:
        pred = self.predict(x)
        return int(np.argmin(pred))


@dataclass
class BitRankerModel:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    device: str

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        xx = _standardize_apply(x.astype(np.float32), self.mean, self.std)
        with torch.no_grad():
            out = self.model(torch.from_numpy(xx).to(self.device)).reshape(-1).cpu().numpy()
        return out.astype(np.float32)

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)



def _fit_common(
    x: np.ndarray,
    y: np.ndarray,
    hidden_dims,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    task: str,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, Dict[str, float]]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D features, got shape={x.shape}")
    x = x.astype(np.float32)
    y = y.astype(np.float32).reshape(-1)
    if len(x) != len(y):
        raise ValueError(f"Feature/label length mismatch: {len(x)} vs {len(y)}")

    mean, std = _standardize_fit(x)
    x_std = _standardize_apply(x, mean, std)

    x_t = torch.from_numpy(x_std)
    y_t = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
    dataset = TensorDataset(x_t, y_t)
    n_total = len(dataset)
    n_val = max(1, int(round(0.1 * n_total))) if n_total > 1 else 1
    n_train = max(n_total - n_val, 1)
    if n_train + n_val > n_total:
        n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1234))

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, n_train), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(n_val, 1)), shuffle=False)

    model = TabularMLP(x.shape[1], hidden_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if task == "regression":
        loss_fn = nn.MSELoss()
    else:
        pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                bs = xb.shape[0]
                val_loss += float(loss.item()) * bs
                n_seen += bs
        val_loss /= max(n_seen, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, mean, std, {
        "best_val_loss": best_val,
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_val": int(n_val),
    }



def fit_snapshot_selector(x, y, cfg, device="cpu"):
    tr_cfg = cfg["training"]
    return _fit_common(
        x=x,
        y=y,
        hidden_dims=tr_cfg["snapshot_hidden_dims"],
        epochs=int(tr_cfg["epochs"]),
        batch_size=int(tr_cfg["batch_size"]),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg["weight_decay"]),
        device=device,
        task="regression",
    )



def fit_bit_ranker(x, y, cfg, device="cpu"):
    tr_cfg = cfg["training"]
    return _fit_common(
        x=x,
        y=y,
        hidden_dims=tr_cfg["bit_hidden_dims"],
        epochs=int(tr_cfg["epochs"]),
        batch_size=int(tr_cfg["batch_size"]),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg["weight_decay"]),
        device=device,
        task="binary",
    )



def save_model_bundle(path: str | Path, model: nn.Module, mean: np.ndarray, std: np.ndarray, meta: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": torch.from_numpy(_to_numpy(mean)),
            "std": torch.from_numpy(_to_numpy(std)),
            "meta": meta,
        },
        path,
    )



def load_snapshot_selector(path: str | Path, hidden_dims, in_dim: int, device: str = "cpu") -> SnapshotSelectorModel:
    ckpt = _safe_torch_load(path, device=device)
    model = TabularMLP(in_dim, hidden_dims).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return SnapshotSelectorModel(
        model=model,
        mean=_to_numpy(ckpt["mean"]),
        std=_to_numpy(ckpt["std"]),
        device=device,
    )



def load_bit_ranker(path: str | Path, hidden_dims, in_dim: int, device: str = "cpu") -> BitRankerModel:
    ckpt = _safe_torch_load(path, device=device)
    model = TabularMLP(in_dim, hidden_dims).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return BitRankerModel(
        model=model,
        mean=_to_numpy(ckpt["mean"]),
        std=_to_numpy(ckpt["std"]),
        device=device,
    )



def save_meta_json(path: str | Path, data: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@dataclass
class ActionPriorModel:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    device: str

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        xx = _standardize_apply(x.astype(np.float32), self.mean, self.std)
        with torch.no_grad():
            out = self.model(torch.from_numpy(xx).to(self.device)).reshape(-1).cpu().numpy()
        return out.astype(np.float32)

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


@dataclass
class StateValueModel:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    device: str

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        xx = _standardize_apply(x.astype(np.float32), self.mean, self.std)
        with torch.no_grad():
            out = self.model(torch.from_numpy(xx).to(self.device)).reshape(-1).cpu().numpy()
        return out.astype(np.float32)

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)





@dataclass
class TemplateRankerModel:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    device: str

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        xx = _standardize_apply(x.astype(np.float32), self.mean, self.std)
        with torch.no_grad():
            out = self.model(torch.from_numpy(xx).to(self.device)).reshape(-1).cpu().numpy()
        return out.astype(np.float32)

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def fit_action_prior(x, y, cfg, device="cpu"):
    tr_cfg = cfg["training"]
    hidden_dims = tr_cfg.get("action_hidden_dims", tr_cfg["bit_hidden_dims"])
    return _fit_common(
        x=x,
        y=y,
        hidden_dims=hidden_dims,
        epochs=int(tr_cfg["epochs"]),
        batch_size=int(tr_cfg["batch_size"]),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg["weight_decay"]),
        device=device,
        task="binary",
    )



def fit_state_value(x, y, cfg, device="cpu"):
    tr_cfg = cfg["training"]
    hidden_dims = tr_cfg.get("value_hidden_dims", tr_cfg["snapshot_hidden_dims"])
    return _fit_common(
        x=x,
        y=y,
        hidden_dims=hidden_dims,
        epochs=int(tr_cfg["epochs"]),
        batch_size=int(tr_cfg["batch_size"]),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg["weight_decay"]),
        device=device,
        task="binary",
    )



def load_action_prior(path: str | Path, hidden_dims, in_dim: int, device: str = "cpu") -> ActionPriorModel:
    ckpt = _safe_torch_load(path, device=device)
    model = TabularMLP(in_dim, hidden_dims).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ActionPriorModel(
        model=model,
        mean=_to_numpy(ckpt["mean"]),
        std=_to_numpy(ckpt["std"]),
        device=device,
    )



def load_state_value(path: str | Path, hidden_dims, in_dim: int, device: str = "cpu") -> StateValueModel:
    ckpt = _safe_torch_load(path, device=device)
    model = TabularMLP(in_dim, hidden_dims).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return StateValueModel(
        model=model,
        mean=_to_numpy(ckpt["mean"]),
        std=_to_numpy(ckpt["std"]),
        device=device,
    )



def fit_template_ranker(x, y, cfg, device="cpu"):
    tr_cfg = cfg["training"]
    hidden_dims = tr_cfg.get("template_hidden_dims", tr_cfg.get("action_hidden_dims", tr_cfg["bit_hidden_dims"]))
    return _fit_common(
        x=x,
        y=y,
        hidden_dims=hidden_dims,
        epochs=int(tr_cfg["epochs"]),
        batch_size=int(tr_cfg["batch_size"]),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg["weight_decay"]),
        device=device,
        task="binary",
    )


def load_template_ranker(path: str | Path, hidden_dims, in_dim: int, device: str = "cpu") -> TemplateRankerModel:
    ckpt = _safe_torch_load(path, device=device)
    model = TabularMLP(in_dim, hidden_dims).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return TemplateRankerModel(
        model=model,
        mean=_to_numpy(ckpt["mean"]),
        std=_to_numpy(ckpt["std"]),
        device=device,
    )
