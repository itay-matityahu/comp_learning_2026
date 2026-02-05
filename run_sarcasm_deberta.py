#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_scheduler
from datasets import Dataset


@dataclass
class TrainConfig:
    model_name: str = "microsoft/deberta-v3-base"
    pooling: str = "mean"  # "cls" or "mean"
    head_hidden_sizes: tuple = (512, 256)
    dropout: float = 0.2
    num_labels: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 2
    train_batch_size: int = 16
    eval_batch_size: int = 32
    max_length: int = 64
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    warmup_ratio: float = 0.06
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = False


class DebertaWithCustomHead(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        head_hidden_sizes=(256,),
        dropout: float = 0.2,
        activation: str = "gelu",
        pooling: str = "cls",
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.pooling = pooling

        hidden_size = self.encoder.config.hidden_size
        layers = []
        in_dim = hidden_size
        act_layer = nn.GELU if activation.lower() == "gelu" else nn.ReLU
        for h in head_hidden_sizes:
            layers += [nn.Linear(in_dim, h), act_layer(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Dropout(dropout), nn.Linear(in_dim, num_labels)]
        self.classifier = nn.Sequential(*layers)

        self.loss_fn = nn.CrossEntropyLoss()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden = out.last_hidden_state

        if self.pooling == "mean":
            pooled = self._mean_pool(last_hidden, attention_mask)
        else:
            pooled = last_hidden[:, 0]

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_headlines(data_dir: Path):
    def read_file(path: Path):
        with path.open("r", encoding="utf-8") as f:
            rows = [line.strip().split(None, 3)[2:] for line in f]
        df = pd.DataFrame(rows, columns=["label", "headline"])
        df["label"] = df["label"].astype(int)
        return df

    train_df = read_file(data_dir / "headline_train.txt")
    val_df = read_file(data_dir / "headline_val.txt")
    test_df = read_file(data_dir / "headline_test.txt")
    return train_df, val_df, test_df


def build_dataloaders(train_df, val_df, test_df, tokenizer, cfg: TrainConfig, device):
    if cfg.max_train_samples is not None:
        train_df = train_df.head(cfg.max_train_samples)
    if cfg.max_eval_samples is not None:
        val_df = val_df.head(cfg.max_eval_samples)
        test_df = test_df.head(cfg.max_eval_samples)
    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch["headline"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )
        tokenized["label"] = [int(l) for l in batch["label"]]
        return tokenized

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["headline"])
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["headline"])
    test_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=["headline"])

    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        attention_mask = torch.tensor([item["attention_mask"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_tok,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_tok,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_tok,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def eval_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["label"]
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = torch.argmax(outputs["logits"], dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def get_preds_labels(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["label"]
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = torch.argmax(outputs["logits"], dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def bootstrap_accuracy(y_pred, y_true, num_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = rng.integers(0, n, size=(num_samples, n))
    accs = (y_pred[indices] == y_true[indices]).mean(axis=1)
    return accs


def train_one_config(train_loader, cfg: TrainConfig, device):
    model = DebertaWithCustomHead(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        head_hidden_sizes=cfg.head_hidden_sizes,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(len(train_loader) / cfg.grad_accum_steps))
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(cfg.num_epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"]

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=labels,
                    )
                    loss = out["loss"] / cfg.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                )
                loss = out["loss"] / cfg.grad_accum_steps
                loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

        print(f"epoch {epoch+1}/{cfg.num_epochs} - train_loss={float(loss.item() * cfg.grad_accum_steps):.4f}")

    return model


def print_random_test_samples(test_df, y_pred, y_true, seed=42, per_class=5):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    idx_0 = np.where(y_true == 0)[0]
    idx_1 = np.where(y_true == 1)[0]
    if len(idx_0) < per_class or len(idx_1) < per_class:
        print("Not enough samples to draw 5 per class.")
        return
    pick_0 = rng.choice(idx_0, size=per_class, replace=False)
    pick_1 = rng.choice(idx_1, size=per_class, replace=False)
    picks = np.concatenate([pick_0, pick_1])
    rng.shuffle(picks)
    table = pd.DataFrame(
        {
            "headline": test_df.iloc[picks]["headline"].tolist(),
            "true_label": y_true[picks].tolist(),
            "pred_label": y_pred[picks].tolist(),
        }
    )
    print("\nRandom test samples (5 sarcastic, 5 non-sarcastic):")
    print(table.to_string(index=False))


def save_artifacts(model, tokenizer, cfg: TrainConfig, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save full model weights (encoder + custom head)
    torch.save(model.state_dict(), save_dir / "model_state.pt")
    # Save tokenizer files
    tokenizer.save_pretrained(save_dir)
    # Save training config for reproducibility
    with (save_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_config(load_dir: Path, fallback: TrainConfig) -> TrainConfig:
    cfg_path = load_dir / "train_config.json"
    if not cfg_path.exists():
        return fallback
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainConfig(**data)


def load_model(load_dir: Path, cfg: TrainConfig, device):
    model = DebertaWithCustomHead(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        head_hidden_sizes=cfg.head_hidden_sizes,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    ).to(device)
    state_path = load_dir / "model_state.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeBERTa sarcasm detector (single config).")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/daniel.nissani/Library/Mobile Documents/com~apple~CloudDocs/דניאל/לימודים ולמידה/אוניברסיטה /תואר שני/Deep Learning/final_project/data",
        help="Path to txt data folder with headline_train/val/test.txt",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_models/deberta_sarcasm",
        help="Directory to save model_state.pt, tokenizer, and train_config.json",
    )
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="mean")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument(
        "--load-dir",
        type=str,
        default=None,
        help="If set, load model_state.pt and skip training.",
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-out",
        type=str,
        default="bootstrap_results.json",
        help="Output JSON filename (relative to save-dir unless absolute)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device()
    print(f"Using device: {device}")

    cfg = TrainConfig(
        model_name=args.model_name,
        pooling=args.pooling,
        num_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        use_amp=args.use_amp,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    data_dir = Path(args.data_dir).expanduser()
    train_df, val_df, test_df = load_headlines(data_dir)

    load_dir = Path(args.load_dir).expanduser() if args.load_dir else None
    if load_dir:
        cfg = load_config(load_dir, cfg)
        tokenizer = AutoTokenizer.from_pretrained(load_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, tokenizer, cfg, device
    )

    if load_dir:
        model = load_model(load_dir, cfg, device)
        print(f"Loaded model from: {load_dir.resolve()}")
    else:
        model = train_one_config(train_loader, cfg, device)
    val_acc = eval_accuracy(model, val_loader, device)
    test_acc = eval_accuracy(model, test_loader, device)
    print(f"VAL accuracy:  {val_acc:.4f}")
    print(f"TEST accuracy: {test_acc:.4f}")
    # Save immediately after training and evaluation, before bootstrap.
    if not load_dir:
        save_artifacts(model, tokenizer, cfg, Path(args.save_dir))
        print(f"Saved model to: {Path(args.save_dir).resolve()}")

    # Bootstrap on test set
    y_pred, y_true = get_preds_labels(model, test_loader, device)
    boot_accs = bootstrap_accuracy(
        y_pred,
        y_true,
        num_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    boot_summary = {
        "num_samples": int(args.bootstrap_samples),
        "seed": int(args.bootstrap_seed),
        "mean_accuracy": float(boot_accs.mean()),
        "std_accuracy": float(boot_accs.std(ddof=1)),
        "ci_2.5": float(np.percentile(boot_accs, 2.5)),
        "ci_97.5": float(np.percentile(boot_accs, 97.5)),
    }
    out_path = Path(args.bootstrap_out)
    if not out_path.is_absolute():
        out_path = Path(args.save_dir) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(boot_summary, f, indent=2)
    print("Bootstrap accuracy:")
    print(
        f"  mean={boot_summary['mean_accuracy']:.4f}, "
        f"std={boot_summary['std_accuracy']:.4f}, "
        f"95% CI=({boot_summary['ci_2.5']:.4f}, {boot_summary['ci_97.5']:.4f})"
    )
    print(f"Bootstrap results saved to: {out_path.resolve()}")

    # Print 10 random samples (5 sarcastic, 5 non-sarcastic)
    print_random_test_samples(test_df, y_pred, y_true, seed=args.bootstrap_seed, per_class=5)


if __name__ == "__main__":
    main()
