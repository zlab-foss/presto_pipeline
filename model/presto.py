# presto_classifier.py
# Ensure relative imports work when run as a script
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from typing import Iterable, Optional, Tuple, Union, Dict
from pathlib import Path
import json
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix

from models import Encoder, _MLPHead

import matplotlib.pyplot as plt
import seaborn as sns


class PrestoClassifier(BaseEstimator, ClassifierMixin):
    """
    Fine-tunes a Presto Encoder + MLP classification head.

    Works with StreamingPixelBatchLoader batches (PixelBatch dataclass) or dict/tuple fallbacks.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        freeze_encoder: bool = True,
        hidden_layer_sizes: Tuple[int, ...] = (256,),
        head_dropout: float = 0.0,
        activation: str = "gelu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_iter: int = 50,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        device: str = "cpu",
        class_weight: Optional[Union[str, dict, torch.Tensor, np.ndarray, Tuple[float, float]]] = None,
        verbose: bool = True,
        random_state: Optional[int] = None,
        pretrained_encoder_path: Optional[str] = "weights/default_model.pt",
        # logging
        log_dir: str = "logs",
        run_name: Optional[str] = None,
    ):
        self.num_classes = int(num_classes)
        self.freeze_encoder = freeze_encoder
        self.hidden_layer_sizes = hidden_layer_sizes
        self.head_dropout = head_dropout
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.device = device
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.pretrained_encoder_path = pretrained_encoder_path

        self.log_dir = str(log_dir)
        self.run_name = run_name

        # runtime
        self.encoder: Optional[Encoder] = None
        self.head: Optional[nn.Module] = None
        self._built = False
        self._criterion = None
        self._optimizer = None
        self._scheduler = None

        self._best_state = None

        # exposed like sklearn
        self.classes_ = list(range(self.num_classes))
        self.loss_curve_ = []
        self.validation_scores_ = []

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _ensure_logs(self) -> Path:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if self.run_name is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            self.run_name = f"run_{ts}"
        run_dir = Path(self.log_dir) / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _log_append_jsonl(self, path: Path, row: dict):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _build(self, sample):
        device = torch.device(self.device)

        if self.encoder is None:
            self.encoder = Encoder()
        self.encoder.to(device)

        # optionally load pretrained encoder weights
        self._maybe_load_pretrained_encoder()

        # freeze or unfreeze
        self.set_freeze_encoder(self.freeze_encoder)

        enc_dim = self.encoder.embedding_size
        self.head = _MLPHead(
            in_features=enc_dim,
            num_classes=self.num_classes,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout=self.head_dropout,
            activation=self.activation,
        ).to(device)

        params: list = []
        if any(p.requires_grad for p in self.encoder.parameters()):
            params.append({"params": [p for p in self.encoder.parameters() if p.requires_grad], "lr": self.learning_rate})
        params.append({"params": self.head.parameters(), "lr": self.learning_rate})

        self._optimizer = optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self._criterion = self._make_criterion()

        # scheduler (created once)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

        self._built = True

    def _make_criterion(self):
        device = torch.device(self.device)
    
        # -------------------------
        # Binary: num_classes == 1 -> BCEWithLogitsLoss
        # -------------------------
        if self.num_classes == 1:
            if self.class_weight is None:
                return nn.BCEWithLogitsLoss()
    
            # Allow dict {0:w_neg, 1:w_pos} or sequence [w_neg, w_pos]
            w_neg, w_pos = None, None
    
            if isinstance(self.class_weight, dict):
                w_neg = float(self.class_weight.get(0, 1.0))
                w_pos = float(self.class_weight.get(1, 1.0))
    
            elif isinstance(self.class_weight, (list, tuple, np.ndarray)) and len(self.class_weight) == 2:
                w_neg, w_pos = float(self.class_weight[0]), float(self.class_weight[1])
    
            elif torch.is_tensor(self.class_weight) and self.class_weight.numel() == 2:
                w_neg, w_pos = float(self.class_weight[0].item()), float(self.class_weight[1].item())
    
            else:
                return nn.BCEWithLogitsLoss()
    
            # pos_weight = w_pos / w_neg
            pos_weight = torch.tensor([w_pos / max(w_neg, 1e-8)], device=device, dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
        # -------------------------
        # Multi-class: CrossEntropyLoss
        # -------------------------
        weight_tensor = None
    
        if self.class_weight is None:
            weight_tensor = None
    
        elif isinstance(self.class_weight, dict):
            weight_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
            for k, v in self.class_weight.items():
                weight_tensor[int(k)] = float(v)
    
        elif torch.is_tensor(self.class_weight):
            weight_tensor = self.class_weight.float()
    
        elif isinstance(self.class_weight, (list, tuple, np.ndarray)):
            weight_tensor = torch.tensor(self.class_weight, dtype=torch.float32)
    
        return nn.CrossEntropyLoss(weight=None if weight_tensor is None else weight_tensor.to(device))


    def set_freeze_encoder(self, freeze: bool = True):
        self.freeze_encoder = freeze
        self.encoder.requires_grad_(not freeze)
        if hasattr(self.encoder, "pos_embed"):
            self.encoder.pos_embed.requires_grad_(False)
        if hasattr(self.encoder, "month_embed"):
            self.encoder.month_embed.requires_grad_(False)

    @staticmethod
    def _batch_unpack(sample, device: torch.device):
        """
        StreamingPixelBatchLoader expected:
          x: (B,12,17), mask: (B,12,17), dw:(B,12), y:(B,), lon:(B,), lat:(B,)

        Returns:
          X, DW, LL(lon,lat), MK, Y, M(=0)
        """
        M = 0

        # PixelBatch-style (dataclass)
        if hasattr(sample, "x") and hasattr(sample, "dw") and hasattr(sample, "y"):
            X = sample.x
            DW = sample.dw
            MK = sample.mask if hasattr(sample, "mask") else None
            lon = sample.lon
            lat = sample.lat
            LL = torch.stack([lon, lat], dim=1)  # (B,2) lon,lat
            Y = sample.y

        # Dict-style
        elif isinstance(sample, dict):
            X = sample.get("x", sample.get("inputs"))
            DW = sample.get("dw", sample.get("dynamic_world"))
            MK = sample.get("mask", None)
            Y = sample.get("y", sample.get("labels"))

            if "latlons" in sample:
                LL = sample["latlons"]
            else:
                lon = sample.get("lon", None)
                lat = sample.get("lat", None)
                if lon is None or lat is None:
                    raise ValueError("Dict batch missing latlons or lon/lat.")
                LL = torch.stack([lon, lat], dim=1)

            M = sample.get("month", 0)

        # Tuple/list fallback: (X, DW, LL, MK, Y, [month])
        elif isinstance(sample, (list, tuple)):
            if len(sample) < 5:
                raise ValueError(f"Expected at least 5 elements (X,DW,LL,MK,Y), got {len(sample)}")
            X, DW, LL, MK, Y = sample[:5]
            M = sample[5] if len(sample) > 5 else 0
        else:
            raise TypeError(f"Unsupported batch type: {type(sample)}")

        X = X.to(device).float()
        DW = DW.to(device).long()
        LL = LL.to(device).float()
        MK = MK.to(device).float() if torch.is_tensor(MK) else None
        Y = Y.to(device).long()

        if torch.is_tensor(M):
            M = M.to(device)

        return X, DW, LL, MK, Y, M

    def _forward_logits(self, X, DW, LL, MK, M):
        z = self.encoder(x=X, dynamic_world=DW, latlons=LL, mask=MK, month=M, eval_task=True)
        logits = self.head(z)
        return logits

    # ---------------------------
    # Public API
    # ---------------------------
    def fit(
        self,
        train_loader,
        val_loader=None,
        best_ckpt_path=None,
    ):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
    
        device = torch.device(self.device)
        run_dir = self._ensure_logs()
        log_path = run_dir / "train_log.jsonl"
    
        first_sample = next(iter(train_loader))
        if not self._built:
            self._build(first_sample)
    
        warmup_epochs = 1
        base_lr = float(self.learning_rate)
        warmup_start_lr = max(base_lr * 0.01, 1e-6)
    
        for pg in self._optimizer.param_groups:
            pg["lr"] = warmup_start_lr
    
        best_val = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
    
        for epoch in range(self.max_iter):
            t0 = time.time()
    
            # warmup lr
            if epoch < warmup_epochs:
                lr = warmup_start_lr + (base_lr - warmup_start_lr) * ((epoch + 1) / warmup_epochs)
                for pg in self._optimizer.param_groups:
                    pg["lr"] = lr
    
            # -----------------------------
            # Training
            # -----------------------------
            self.encoder.train(not self.freeze_encoder)
            self.head.train()
    
            running_loss = 0.0
            n_batches = 0
    
            # exponential moving average for display (less noisy)
            ema_loss = None
            ema_beta = 0.98
    
            train_bar = tqdm(
                train_loader,
                desc=f"Train [{epoch+1}/{self.max_iter}]",
                leave=False,
                disable=not self.verbose,
            )
    
            for sample in train_bar:
                X, DW, LL, MK, Y, M = self._batch_unpack(sample, device)
    
                self._optimizer.zero_grad(set_to_none=True)
                logits = self._forward_logits(X, DW, LL, MK, M)
    
                if self.num_classes == 1:
                    loss = self._criterion(logits.squeeze(1), Y.float())
                else:
                    loss = self._criterion(logits, Y.long())
    
                loss.backward()
                self._optimizer.step()
    
                loss_val = float(loss.item())
                running_loss += loss_val
                n_batches += 1
    
                # update EMA for prettier tqdm
                if ema_loss is None:
                    ema_loss = loss_val
                else:
                    ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss_val
    
                lr_now = float(self._optimizer.param_groups[0]["lr"])
    
                # show live loss
                train_bar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    ema=f"{ema_loss:.4f}",
                    avg=f"{(running_loss / n_batches):.4f}",
                    lr=f"{lr_now:.2e}",
                )
    
            avg_train = running_loss / max(1, n_batches)
            self.loss_curve_.append(avg_train)
    
            # -----------------------------
            # Validation
            # -----------------------------
            avg_val = None
            if val_loader is not None:
                self.encoder.eval()
                self.head.eval()
    
                val_loss = 0.0
                vb = 0
    
                val_bar = tqdm(
                    val_loader,
                    desc=f"Val   [{epoch+1}/{self.max_iter}]",
                    leave=False,
                    disable=not self.verbose,
                )
    
                with torch.no_grad():
                    for sample in val_bar:
                        X, DW, LL, MK, Y, M = self._batch_unpack(sample, device)
                        logits = self._forward_logits(X, DW, LL, MK, M)
    
                        if self.num_classes == 1:
                            loss = self._criterion(logits.squeeze(1), Y.float())
                        else:
                            loss = self._criterion(logits, Y.long())
    
                        lv = float(loss.item())
                        val_loss += lv
                        vb += 1
    
                        val_bar.set_postfix(
                            loss=f"{lv:.4f}",
                            avg=f"{(val_loss / vb):.4f}",
                        )
    
                avg_val = val_loss / max(1, vb)
                self.validation_scores_.append(avg_val)
    
                # step scheduler only after warmup
                if epoch >= warmup_epochs:
                    self._scheduler.step(avg_val)
    
                # best ckpt + early stop
                if avg_val < best_val - 1e-8:
                    best_val = avg_val
                    best_epoch = epoch
                    epochs_no_improve = 0
    
                    self._best_state = {
                        "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
                        "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
                    }
    
                    if best_ckpt_path is not None:
                        Path(best_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                        self.save(best_ckpt_path)
                        if self.verbose:
                            print(f"💾 Saved BEST checkpoint @ epoch {epoch+1} (val={best_val:.6f}) -> {best_ckpt_path}")
                else:
                    epochs_no_improve += 1
                    if self.early_stopping and epochs_no_improve >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1} (best epoch={best_epoch+1}, best val={best_val:.6f})")
    
                        lr_now = float(self._optimizer.param_groups[0]["lr"])
                        self._log_append_jsonl(log_path, {
                            "epoch": epoch + 1,
                            "train_loss": avg_train,
                            "val_loss": avg_val,
                            "lr": lr_now,
                            "seconds": time.time() - t0,
                            "early_stop": True,
                        })
                        break
    
            lr_now = float(self._optimizer.param_groups[0]["lr"])
            if self.verbose:
                if avg_val is None:
                    print(f"Epoch {epoch+1}/{self.max_iter} - loss: {avg_train:.4f} - lr: {lr_now:.2e}")
                else:
                    print(f"Epoch {epoch+1}/{self.max_iter} - loss: {avg_train:.4f} - val: {avg_val:.4f} - lr: {lr_now:.2e}")
    
            self._log_append_jsonl(log_path, {
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "lr": lr_now,
                "seconds": time.time() - t0,
                "warmup": epoch < warmup_epochs,
                "best_val": best_val,
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
            })
    
        if self._best_state is not None:
            self.encoder.load_state_dict(self._best_state["encoder"])
            self.head.load_state_dict(self._best_state["head"])
    
        try:
            self._save_training_plots(run_dir)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Could not save plots: {e}")
    
        return self


    def _save_training_plots(self, run_dir: Path):
        # loss curves
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_curve_, label="train_loss")
        if len(self.validation_scores_) > 0:
            plt.plot(self.validation_scores_, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training curves")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss_curves.png", dpi=150)
        plt.close()

        # also save as csv
        import pandas as pd
        n = max(len(self.loss_curve_), len(self.validation_scores_))
        rows = []
        for i in range(n):
            rows.append({
                "epoch": i + 1,
                "train_loss": self.loss_curve_[i] if i < len(self.loss_curve_) else None,
                "val_loss": self.validation_scores_[i] if i < len(self.validation_scores_) else None,
            })
        pd.DataFrame(rows).to_csv(run_dir / "loss_curves.csv", index=False)

    # ---------------- inference / eval ----------------
    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        self.encoder.eval()
        self.head.eval()
        device = torch.device(self.device)
        preds = []
        for sample in tqdm(data_loader, desc="Predict"):
            X, DW, LL, MK, _, M = self._batch_unpack(sample, device)
            logits = self._forward_logits(X, DW, LL, MK, M)
            if self.num_classes == 1:
                y = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
            else:
                y = torch.argmax(logits, dim=1).long()
            preds.append(y.cpu().numpy())
        return np.concatenate(preds, axis=0)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, class_names: Optional[Iterable[str]] = None, verbose: int = 0):
        device = torch.device(self.device)
        self.encoder.eval()
        self.head.eval()
        all_preds, all_labels = [], []
        for sample in tqdm(data_loader, desc="Evaluate"):
            X, DW, LL, MK, Y, M = self._batch_unpack(sample, device)
            logits = self._forward_logits(X, DW, LL, MK, M)
            if self.num_classes == 1:
                yhat = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
            else:
                yhat = torch.argmax(logits, dim=1).long()
            all_preds.append(yhat.cpu().numpy())
            all_labels.append(Y.long().cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        if class_names is None:
            class_names = [str(c) for c in range(self.num_classes)]

        acc = float((y_pred == y_true).mean())
        report = classification_report(
            y_true, y_pred,
            labels=list(range(self.num_classes)),
            target_names=list(class_names),
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        if verbose >= 1:
            print(f"\nAccuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(
                y_true, y_pred,
                labels=list(range(self.num_classes)),
                target_names=list(class_names),
                zero_division=0
            ))
        if verbose >= 2:
            cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)), normalize="true") * 100
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                        xticklabels=list(class_names), yticklabels=list(class_names))
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Normalized Confusion Matrix (Acc: {acc:.2%})")
            plt.tight_layout()
            plt.show()

        return {"accuracy": acc, "classification_report": report, "confusion_matrix": cm}

    # ---------------- persistence ----------------
    def save(self, filepath: str):
        ckpt = {
            "encoder_state": self.encoder.state_dict(),
            "head_state": self.head.state_dict(),
            "freeze_encoder": self.freeze_encoder,
            "num_classes": self.num_classes,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "head_dropout": self.head_dropout,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "device": self.device,
            "class_weight": self.class_weight,
            "pretrained_encoder_path": self.pretrained_encoder_path,
            "log_dir": self.log_dir,
            "run_name": self.run_name,
        }
        if self._optimizer is not None:
            ckpt["optimizer_state"] = self._optimizer.state_dict()
        torch.save(ckpt, filepath)

    # ------- PRETRAINED ENCODER LOADING -------
    def load_pretrained_encoder(self, ckpt_path: str = "weights/default_model.pt", *, strict: bool = False, verbose: bool = True):
        device = torch.device(self.device)
        ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "encoder_state_dict" in ckpt and isinstance(ckpt["encoder_state_dict"], dict):
                state = ckpt["encoder_state_dict"]
            elif "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
                state = ckpt["encoder"]
            else:
                state = ckpt
        else:
            state = ckpt

        enc_state = {}
        for k, v in state.items():
            if k.startswith("module.encoder."):
                enc_state[k[len("module.encoder."):]] = v
            elif k.startswith("encoder."):
                enc_state[k[len("encoder."):]] = v
            else:
                if self.encoder is not None and (k in self.encoder.state_dict()):
                    enc_state[k] = v

        if len(enc_state) == 0:
            raise RuntimeError(f"No encoder weights found in checkpoint: {ckpt_path}")

        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=strict)
        if verbose and self.verbose:
            if missing:
                print(f"[PrestoClassifier] Missing encoder keys ({len(missing)}): {sorted(missing)[:8]}{' ...' if len(missing)>8 else ''}")
            if unexpected:
                print(f"[PrestoClassifier] Unexpected encoder keys ({len(unexpected)}): {sorted(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")
            if not missing and not unexpected:
                print("[PrestoClassifier] Pretrained encoder loaded cleanly.")
        return self

    def _maybe_load_pretrained_encoder(self):
        path = getattr(self, "pretrained_encoder_path", None)
        if path:
            self.load_pretrained_encoder(path, strict=False, verbose=self.verbose)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "PrestoClassifier":
        """
        Load a saved PrestoClassifier checkpoint AND restore training attributes/metadata.
    
        Restores:
          - model init hyperparams (from checkpoint)
          - encoder/head state
          - optimizer state (if present)
          - scheduler state (if present)
          - training curves (loss_curve_, validation_scores_) if present
          - best tracking (_best_val, _best_state, _epochs_no_improve) if present
          - logging config (log_dir, run_name) if present
    
        Notes:
          - Uses a dummy batch to build modules.
          - Keeps sklearn-style attributes.
        """
        ckpt = torch.load(filepath, map_location=device)
    
        # ---- construct model from saved hyperparams ----
        model = cls(
            num_classes=ckpt["num_classes"],
            freeze_encoder=ckpt.get("freeze_encoder", True),
            hidden_layer_sizes=tuple(ckpt.get("hidden_layer_sizes", (256,))),
            head_dropout=float(ckpt.get("head_dropout", 0.0)),
            activation=ckpt.get("activation", "gelu"),
            learning_rate=float(ckpt.get("learning_rate", 1e-3)),
            weight_decay=float(ckpt.get("weight_decay", 1e-4)),
            max_iter=int(ckpt.get("max_iter", 50)),
            early_stopping=bool(ckpt.get("early_stopping", False)),
            n_iter_no_change=int(ckpt.get("n_iter_no_change", 10)),
            device=device,
            class_weight=ckpt.get("class_weight", None),
            pretrained_encoder_path=ckpt.get("pretrained_encoder_path", None),
    
            # NEW: restore logging params if your __init__ supports them
            log_dir=ckpt.get("log_dir", "logs"),
            run_name=ckpt.get("run_name", None),
            verbose=ckpt.get("verbose", True),
            random_state=ckpt.get("random_state", None),
        )
    
        # ---- build modules ----
        dummy = {
            "inputs": torch.zeros(1, 2, 17),
            "dynamic_world": torch.zeros(1, 2, dtype=torch.long),
            "latlons": torch.zeros(1, 2),
            "mask": torch.zeros(1, 2, 17),
            "labels": torch.zeros(1, dtype=torch.long),
        }
        model._build(dummy)
    
        # ---- restore weights ----
        if "encoder_state" in ckpt:
            model.encoder.load_state_dict(ckpt["encoder_state"])
        elif "encoder_state_dict" in ckpt:
            model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        else:
            raise KeyError("Checkpoint missing encoder_state.")
    
        if "head_state" in ckpt:
            model.head.load_state_dict(ckpt["head_state"])
        elif "head_state_dict" in ckpt:
            model.head.load_state_dict(ckpt["head_state_dict"])
        else:
            raise KeyError("Checkpoint missing head_state.")
    
        # ---- restore optimizer ----
        if ckpt.get("optimizer_state", None) is not None and model._optimizer is not None:
            try:
                model._optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                # optimizer groups might differ if freeze/unfreeze changed; ignore safely
                if getattr(model, "verbose", True):
                    print(f"[WARN] Could not load optimizer_state: {e}")
    
        # ---- restore scheduler (optional) ----
        if ckpt.get("scheduler_state", None) is not None and hasattr(model, "_scheduler") and model._scheduler is not None:
            try:
                model._scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception as e:
                if getattr(model, "verbose", True):
                    print(f"[WARN] Could not load scheduler_state: {e}")
    
        # ---- restore training curves / bookkeeping (optional) ----
        model.loss_curve_ = list(ckpt.get("loss_curve_", []))
        model.validation_scores_ = list(ckpt.get("validation_scores_", []))
    
        model._best_val = float(ckpt.get("_best_val", float("inf")))
        model._epochs_no_improve = int(ckpt.get("_epochs_no_improve", 0))
    
        # restore best_state if present (CPU tensors)
        if ckpt.get("_best_state", None) is not None:
            model._best_state = ckpt["_best_state"]
        else:
            model._best_state = None
    
        # restore classes_ if present (sklearn-style)
        if ckpt.get("classes_", None) is not None:
            model.classes_ = ckpt["classes_"]
        else:
            model.classes_ = list(range(model.num_classes))
    
        # mark built
        model._built = True
    
        return model
