# Ensure relative imports work when run as a script
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


from typing import Iterable, Optional, Tuple, Union, Dict
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

    Expected batch from DataLoader:
        sample = {
            "inputs":         X,   # float tensor [B, T, C=17] (see presto Encoder band groups)
            "dynamic_world":  DW,  # long  tensor [B, T]  (0..8, 9 for missing)
            "latlons":        LL,  # float tensor [B, 2]  (lat, lon in degrees)
            "mask":           MK,  # float tensor [B, T, C] with 1 for masked/invalid tokens
            "labels":         Y,   # long  tensor [B] for multi-class; or float [B] for binary
            # Optional: "month":  int or tensor
        }

    Parameters
    ----------
    num_classes : int
        Number of output classes. Set 1 for binary (logit).
    freeze_encoder : bool, default=True
        If True, encoder weights are frozen during training.
    hidden_layer_sizes : tuple, default=(256,)
        MLP head hidden sizes.
    head_dropout : float, default=0.0
        Dropout probability in head.
    activation : str, default='gelu'
        Activation for head layers.
    learning_rate : float, default=1e-3
    weight_decay : float, default=1e-4
    max_iter : int, default=50
        Epochs.
    early_stopping : bool, default=False
    n_iter_no_change : int, default=10
    device : str, default='cpu'
    class_weight : None | 'balanced' | dict | tensor | array-like
        Per-class weights for CrossEntropy (multi-class) or pos_weight for BCE (binary).
        - For multi-class: behaves like PyTorch `weight=` in CrossEntropyLoss.
        - For binary with num_classes==1: if dict/array provided as [w_neg, w_pos],
          internally uses BCEWithLogitsLoss(pos_weight=w_pos / w_neg).
    verbose : bool, default=True
    random_state : Optional[int], default=None
    pretrained_encoder_path : Optional[str], default="weights/default_model.pt"
        If not None, will load encoder weights from this checkpoint during build.
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

        # runtime
        self.encoder: Optional[Encoder] = None
        self.head: Optional[nn.Module] = None
        self._built = False
        self._criterion = None
        self._optimizer = None
        self._best_val = float("inf")
        self._epochs_no_improve = 0

        # exposed like sklearn
        self.classes_ = list(range(self.num_classes))
        self.loss_curve_ = []
        self.validation_scores_ = []

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _build(self, sample: Dict[str, torch.Tensor]):
        device = torch.device(self.device)

        # encoder
        if self.encoder is None:
            self.encoder = Encoder()
        self.encoder.to(device)

        # optionally load pretrained encoder weights
        self._maybe_load_pretrained_encoder()

        # freeze or unfreeze
        self.set_freeze_encoder(self.freeze_encoder)

        # head
        enc_dim = self.encoder.embedding_size
        self.head = _MLPHead(
            in_features=enc_dim,
            num_classes=self.num_classes,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout=self.head_dropout,
            activation=self.activation,
        ).to(device)

        # optimizer (separate LR groups if encoder is unfrozen)
        params: list = []
        if any(p.requires_grad for p in self.encoder.parameters()):
            params.append({"params": [p for p in self.encoder.parameters() if p.requires_grad],
                           "lr": self.learning_rate})
        params.append({"params": self.head.parameters(), "lr": self.learning_rate})

        self._optimizer = optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # criterion
        self._criterion = self._make_criterion()

        self._built = True

    def _make_criterion(self):
        # Binary: num_classes == 1 -> BCEWithLogits
        if self.num_classes == 1:
            if self.class_weight is None:
                return nn.BCEWithLogitsLoss()
            # allow [w_neg, w_pos] or dict {0:w_neg,1:w_pos}
            if isinstance(self.class_weight, dict):
                w_neg = float(self.class_weight.get(0, 1.0))
                w_pos = float(self.class_weight.get(1, 1.0))
            elif isinstance(self.class_weight, (list, tuple, np.ndarray)) and len(self.class_weight) == 2:
                w_neg, w_pos = float(self.class_weight[0]), float(self.class_weight[1])
            elif torch.is_tensor(self.class_weight) and self.class_weight.numel() == 2:
                w_neg, w_pos = float(self.class_weight[0].item()), float(self.class_weight[1].item())
            else:
                # fallback
                return nn.BCEWithLogitsLoss()
            pos_weight = torch.tensor([w_pos / max(w_neg, 1e-8)], device=self.device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Multi-class
        else:
            weight_tensor = None
            if isinstance(self.class_weight, dict):
                weight_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
                for k, v in self.class_weight.items():
                    weight_tensor[int(k)] = float(v)
            elif torch.is_tensor(self.class_weight):
                weight_tensor = self.class_weight.float()
            elif isinstance(self.class_weight, (list, tuple, np.ndarray)):
                weight_tensor = torch.tensor(self.class_weight, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=None if weight_tensor is None else weight_tensor.to(self.device))

    def set_freeze_encoder(self, freeze: bool = True):
        self.freeze_encoder = freeze
        self.encoder.requires_grad_(not freeze)
        # keep these frozen per your encoder design
        if hasattr(self.encoder, "pos_embed"):
            self.encoder.pos_embed.requires_grad_(False)
        if hasattr(self.encoder, "month_embed"):
            self.encoder.month_embed.requires_grad_(False)

    @staticmethod
    def _to_device(x, device):
        if x is None:
            return None
        return x.to(device) if torch.is_tensor(x) else x

    @staticmethod
    def _batch_unpack(sample, device: torch.device):
        """
        Accepts either:
          - dict with keys: inputs, dynamic_world, latlons, mask, labels, (optional) month
          - tuple/list:      (X, DW, LL, MK, Y, [month])
        Moves tensors to device and returns: X, DW, LL, MK, Y, M
        """
        # Dict-style dataset
        if isinstance(sample, dict):
            X  = sample["inputs"]
            DW = sample["dynamic_world"]
            LL = sample["latlons"]
            MK = sample.get("mask", None)
            Y  = sample["labels"]
            M  = sample.get("month", 0)

        # Tuple/list-style dataset
        elif isinstance(sample, (list, tuple)):
            if len(sample) < 5:
                raise ValueError(
                    f"Expected at least 5 elements (X, DW, LL, MK, Y), got {len(sample)}"
                )
            X, DW, LL, MK, Y = sample[:5]
            M = sample[5] if len(sample) > 5 else 0
        else:
            raise TypeError(f"Unsupported batch type: {type(sample)}")

        # To device + dtype normalization
        X  = X.to(device)                          # [B, T, C]
        DW = DW.to(device).long()                  # [B, T]  ensure long
        LL = LL.to(device).float()                 # [B, 2]
        MK = MK.to(device).float() if torch.is_tensor(MK) else None
        Y  = Y.to(device)                          # [B] (long for CE) or float for BCE
        if torch.is_tensor(M):
            M = M.to(device)
        # If month is an int, leave it as-is; Encoder handles both int/tensor
        return X, DW, LL, MK, Y, M


    def _forward_logits(self, X, DW, LL, MK, M):
        # Encoder returns pooled embedding [B, D] when eval_task=True
        z = self.encoder(x=X, dynamic_world=DW, latlons=LL, mask=MK, month=M, eval_task=True)
        logits = self.head(z)  # [B, num_classes] or [B,1]
        return logits

    # ---------------------------
    # Public API
    # ---------------------------
    # def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
    #     if not isinstance(train_loader, DataLoader):
    #         raise TypeError("train_loader must be a PyTorch DataLoader")
    #     if self.random_state is not None:
    #         torch.manual_seed(self.random_state)
    #         np.random.seed(self.random_state)

    #     device = torch.device(self.device)

    #     # Build on first batch (to access encoder embedding size etc.)
    #     first_sample = next(iter(train_loader))
    #     if not self._built:
    #         self._build(first_sample)

    #     best_val = float("inf")
    #     epochs_no_improve = 0

    #     for epoch in range(self.max_iter):
    #         self.encoder.train(not self.freeze_encoder)
    #         self.head.train()

    #         running_loss = 0.0
    #         n_batches = 0

    #         for sample in train_loader:
    #             X, DW, LL, MK, Y, M = self._batch_unpack(sample, device)

    #             self._optimizer.zero_grad()
    #             logits = self._forward_logits(X, DW, LL, MK, M)

    #             if self.num_classes == 1:
    #                 # expect Y in {0,1} float
    #                 if Y.dtype != torch.float32:
    #                     Y = Y.float()
    #                 loss = self._criterion(logits.squeeze(1), Y)
    #             else:
    #                 # expect Y in long indices
    #                 if Y.dtype != torch.long:
    #                     Y = Y.long()
    #                 loss = self._criterion(logits, Y)

    #             loss.backward()
    #             self._optimizer.step()

    #             running_loss += float(loss.item())
    #             n_batches += 1

    #         avg_train = running_loss / max(1, n_batches)
    #         self.loss_curve_.append(avg_train)

    #         # validation
    #         if val_loader is not None:
    #             self.encoder.eval()
    #             self.head.eval()
    #             val_loss = 0.0
    #             vb = 0
    #             with torch.no_grad():
    #                 for sample in val_loader:
    #                     X, DW, LL, MK, Y, M = self._batch_unpack(sample, device)
    #                     logits = self._forward_logits(X, DW, LL, MK, M)
    #                     if self.num_classes == 1:
    #                         if Y.dtype != torch.float32:
    #                             Y = Y.float()
    #                         loss = self._criterion(logits.squeeze(1), Y)
    #                     else:
    #                         if Y.dtype != torch.long:
    #                             Y = Y.long()
    #                         loss = self._criterion(logits, Y)
    #                     val_loss += float(loss.item())
    #                     vb += 1
    #             avg_val = val_loss / max(1, vb)
    #             self.validation_scores_.append(avg_val)

    #             if self.verbose:
    #                 print(f"Epoch {epoch+1}/{self.max_iter} - loss: {avg_train:.4f} - val: {avg_val:.4f}")

    #             if self.early_stopping:
    #                 if avg_val < best_val - 1e-8:
    #                     best_val = avg_val
    #                     epochs_no_improve = 0
    #                     # keep a lightweight best copy of head (and encoder if unfrozen)
    #                     self._best_state = {
    #                         "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
    #                         "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
    #                     }
    #                 else:
    #                     epochs_no_improve += 1
    #                     if epochs_no_improve >= self.n_iter_no_change:
    #                         if self.verbose:
    #                             print(f"Early stopping at epoch {epoch+1}")
    #                         break
    #         else:
    #             if self.verbose:
    #                 print(f"Epoch {epoch+1}/{self.max_iter} - loss: {avg_train:.4f}")

    #     # restore best (if tracked)
    #     if hasattr(self, "_best_state"):
    #         self.encoder.load_state_dict(self._best_state["encoder"])
    #         self.head.load_state_dict(self._best_state["head"])

    #     return self
    
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader must be a PyTorch DataLoader")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
    
        device = torch.device(self.device)
    
        # Build on first batch
        first_sample = next(iter(train_loader))
        if not self._built:
            self._build(first_sample)
    
        # -----------------------------
        # Prepare warmup
        # -----------------------------
        warmup_epochs = getattr(self, "warmup_epochs", 3)
        base_lr = self.learning_rate
        for pg in self._optimizer.param_groups:
            pg["lr"] = 1e-8   # start near zero
    
        # -----------------------------
        # Prepare scheduler AFTER warmup
        # -----------------------------
        if not hasattr(self, "_scheduler") or self._scheduler is None:
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )
    
        best_val = float("inf")
        epochs_no_improve = 0
    
        for epoch in range(self.max_iter):
    
            # -----------------------------
            # Warmup LR update
            # -----------------------------
            if epoch < warmup_epochs:
                warmup_lr = (epoch + 1) / warmup_epochs * base_lr
                for pg in self._optimizer.param_groups:
                    pg["lr"] = warmup_lr
            # otherwise scheduler controls LR
    
            # -----------------------------
            # Training phase
            # -----------------------------
            self.encoder.train(not self.freeze_encoder)
            self.head.train()
    
            running_loss = 0.0
            n_batches = 0
    
            for sample in train_loader:
                X, DW, LL, MK, Y, M = self._batch_unpack(sample, device)
    
                self._optimizer.zero_grad()
                logits = self._forward_logits(X, DW, LL, MK, M)
    
                if self.num_classes == 1:
                    if Y.dtype != torch.float32:
                        Y = Y.float()
                    loss = self._criterion(logits.squeeze(1), Y)
                else:
                    if Y.dtype != torch.long:
                        Y = Y.long()
                    loss = self._criterion(logits, Y)
    
                loss.backward()
                self._optimizer.step()
    
                running_loss += float(loss.item())
                n_batches += 1
    
            avg_train = running_loss / max(1, n_batches)
            self.loss_curve_.append(avg_train)
    
            # -----------------------------
            # Validation phase
            # -----------------------------
            if val_loader is not None:
                self.encoder.eval()
                self.head.eval()
                val_loss = 0.0
                vb = 0
                with torch.no_grad():
                    for sample in val_loader:
                        X, DW, LL, MK, Y, M = self._batch_unpack(sample, device)
                        logits = self._forward_logits(X, DW, LL, MK, M)
    
                        if self.num_classes == 1:
                            if Y.dtype != torch.float32:
                                Y = Y.float()
                            loss = self._criterion(logits.squeeze(1), Y)
                        else:
                            if Y.dtype != torch.long:
                                Y = Y.long()
                            loss = self._criterion(logits, Y)
    
                        val_loss += float(loss.item())
                        vb += 1
    
                avg_val = val_loss / max(1, vb)
                self.validation_scores_.append(avg_val)
    
                # Only step scheduler AFTER warmup
                if epoch >= warmup_epochs:
                    self._scheduler.step(avg_val)
    
                if self.verbose:
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch+1}/{self.max_iter} "
                        f"- loss: {avg_train:.4f} - val: {avg_val:.4f} "
                        f"- lr: {current_lr:.2e}"
                    )
    
                # -------------------------
                # Early stopping
                # -------------------------
                if self.early_stopping:
                    if avg_val < best_val - 1e-8:
                        best_val = avg_val
                        epochs_no_improve = 0
                        self._best_state = {
                            "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
                            "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
                        }
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.n_iter_no_change:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
    
            else:
                # No validation loader
                if self.verbose:
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch+1}/{self.max_iter} "
                        f"- loss: {avg_train:.4f} "
                        f"- lr: {current_lr:.2e}"
                    )
    
        # ---------------------------------
        # Restore best model (if any)
        # ---------------------------------
        if hasattr(self, "_best_state"):
            self.encoder.load_state_dict(self._best_state["encoder"])
            self.head.load_state_dict(self._best_state["head"])
    
        return self


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
    def predict_proba(self, data_loader: DataLoader) -> np.ndarray:
        self.encoder.eval()
        self.head.eval()
        device = torch.device(self.device)
        probs = []
        for sample in tqdm(data_loader, desc="Predict proba"):
            X, DW, LL, MK, _, M = self._batch_unpack(sample, device)
            logits = self._forward_logits(X, DW, LL, MK, M)
            if self.num_classes == 1:
                p = torch.sigmoid(logits.squeeze(1)).unsqueeze(1)  # [B,1]
                p = torch.cat([1 - p, p], dim=1)                   # [B,2] for compatibility
            else:
                p = torch.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())
        return np.concatenate(probs, axis=0)

    @torch.no_grad()
    def score(self, data_loader: DataLoader) -> float:
        # accuracy
        correct, total = 0, 0
        device = torch.device(self.device)
        self.encoder.eval()
        self.head.eval()
        for sample in tqdm(data_loader, desc="Score"):
            X, DW, LL, MK, Y, M = self._batch_unpack(sample, device)
            logits = self._forward_logits(X, DW, LL, MK, M)
            if self.num_classes == 1:
                yhat = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
            else:
                yhat = torch.argmax(logits, dim=1).long()
            total += Y.size(0)
            correct += (yhat == Y.long()).sum().item()
        return correct / max(1, total)

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

        acc = (y_pred == y_true).mean()
        report = classification_report(y_true, y_pred, labels=list(range(self.num_classes)),
                                       target_names=list(class_names), output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

   
        # ---- Verbose behavior ----
        if verbose >= 1:
            print(f"\nAccuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, labels=list(range(self.num_classes)),
                                        target_names=list(class_names), zero_division=0))
        
        if verbose >= 2:
            # Compute row-normalized CM as percentages for visualization
            cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)), normalize='true') * 100
            
            plt.figure(figsize=(10, 8))  # Slightly larger for better readability
            sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': '%'},
                        xticklabels=list(class_names), yticklabels=list(class_names))
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.title(f"Normalized Confusion Matrix (Accuracy: {acc:.2%})")
            plt.tight_layout()
            plt.show()    
        
        return {"accuracy": acc, "classification_report": report, "confusion_matrix": cm}

    # ------------- persistence -------------
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
        }
        if self._optimizer is not None:
            ckpt["optimizer_state"] = self._optimizer.state_dict()
        torch.save(ckpt, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "PrestoClassifier":
        ckpt = torch.load(filepath, map_location=device)
        model = cls(
            num_classes=ckpt["num_classes"],
            freeze_encoder=ckpt["freeze_encoder"],
            hidden_layer_sizes=tuple(ckpt["hidden_layer_sizes"]),
            head_dropout=ckpt["head_dropout"],
            activation=ckpt["activation"],
            learning_rate=ckpt["learning_rate"],
            weight_decay=ckpt["weight_decay"],
            max_iter=ckpt["max_iter"],
            early_stopping=ckpt["early_stopping"],
            n_iter_no_change=ckpt["n_iter_no_change"],
            device=device,
            class_weight=ckpt["class_weight"],
            pretrained_encoder_path=ckpt.get("pretrained_encoder_path", None),
        )
        # build modules
        dummy = {
            "inputs": torch.zeros(1, 2, 17),
            "dynamic_world": torch.zeros(1, 2, dtype=torch.long),
            "latlons": torch.zeros(1, 2),
            "mask": torch.zeros(1, 2, 17),
            "labels": torch.zeros(1, dtype=torch.long),
        }
        model._build(dummy)
        model.encoder.load_state_dict(ckpt["encoder_state"])
        model.head.load_state_dict(ckpt["head_state"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            model._optimizer.load_state_dict(ckpt["optimizer_state"])
        return model

    # ------- PRETRAINED ENCODER LOADING -------

    def load_pretrained_encoder(
        self,
        ckpt_path: str = "weights/default_model.pt",
        *,
        strict: bool = False,
        verbose: bool = True,
    ):
        """
        Load ONLY the encoder weights from a Presto-style checkpoint.

        Supported layouts:
          - flat state_dict with keys like 'encoder.xxx'
          - DataParallel: 'module.encoder.xxx'
          - nested: {'state_dict': {...}}, {'encoder_state_dict': {...}}, {'encoder': {...}}
          - already-trimmed to encoder keys

        Args:
            ckpt_path: path to .pt file (default: weights/default_model.pt)
            strict: passed to load_state_dict
            verbose: print missing/unexpected keys
        """
        device = torch.device(self.device)
        ckpt = torch.load(ckpt_path, map_location=device)

        # Unwrap common containers
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "encoder_state_dict" in ckpt and isinstance(ckpt["encoder_state_dict"], dict):
                state = ckpt["encoder_state_dict"]
            elif "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
                state = ckpt["encoder"]
            else:
                state = ckpt  # might already be a flat dict
        else:
            state = ckpt

        # Collect only encoder.* weights and strip the prefix
        enc_state = {}
        for k, v in state.items():
            if k.startswith("module.encoder."):
                enc_state[k[len("module.encoder."):]] = v
            elif k.startswith("encoder."):
                enc_state[k[len("encoder."):]] = v
            else:
                # If user already saved trimmed encoder dict (no prefix), accept if it matches any encoder key
                if self.encoder is not None and (k in self.encoder.state_dict()):
                    enc_state[k] = v

        if len(enc_state) == 0:
            raise RuntimeError(
                f"No encoder weights found in checkpoint: {ckpt_path}. "
                "Expected keys starting with 'encoder.' or 'module.encoder.'."
            )

        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=strict)
        if verbose:
            if missing:
                print(f"[PrestoClassifier] Missing encoder keys ({len(missing)}): {sorted(missing)[:8]}{' ...' if len(missing)>8 else ''}")
            if unexpected:
                print(f"[PrestoClassifier] Unexpected encoder keys ({len(unexpected)}): {sorted(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")
            if not missing and not unexpected:
                print("[PrestoClassifier] Pretrained encoder loaded cleanly.")

        return self

    def _maybe_load_pretrained_encoder(self):
        """Internal hook: load encoder if user set `pretrained_encoder_path`."""
        path = getattr(self, "pretrained_encoder_path", None)
        if path:
            self.load_pretrained_encoder(path, strict=False, verbose=self.verbose)
