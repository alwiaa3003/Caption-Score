"""Phase 4b: fine-tune DistilBERT on captions. Meant for a GPU (Google Colab T4); do NOT run
on this project's CPU-only dev machine.

Protocol (same rules as the other models):
  - train on the exported train subsample, EXCLUDING the last 3 training months
    (2025-06..2025-08), which are the inner hold-out used to pick the best checkpoint
  - validation is only predicted, never used for training or checkpoint selection
  - output: val_predictions_distilbert.parquet (video_id, prob) + run_info.json

Usage (Colab):
  python finetune_transformer.py --data captionscore_finetune.parquet --out out
  python finetune_transformer.py ... --shuffle-labels --max-train-rows 30000  # sanity check
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

INNER_MONTHS = ["2025-06", "2025-07", "2025-08"]


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def positive_prob(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return (p / p.sum(axis=1, keepdims=True))[:, 1]


def compute_metrics(pred):
    prob = positive_prob(pred.predictions)
    return {"pr_auc": float(average_precision_score(pred.label_ids, prob))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--shuffle-labels", action="store_true", help="sanity check: expect ~0.10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="only for a tiny local smoke test of the code path")
    args = ap.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    assert set(df["split"]) <= {"train", "val"}, "test/stress must not be in the Colab export"
    train = df[(df["split"] == "train") & ~df["month"].isin(INNER_MONTHS)]
    inner = df[(df["split"] == "train") & df["month"].isin(INNER_MONTHS)]
    val = df[df["split"] == "val"]
    if args.max_train_rows:
        train = train.sample(n=min(args.max_train_rows, len(train)), random_state=args.seed)
    y_train = train["viral"].to_numpy()
    if args.shuffle_labels:
        y_train = np.random.default_rng(args.seed).permutation(y_train)
    print(f"train {len(train):,} | inner hold-out {len(inner):,} | val {len(val):,}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    ds_train = CaptionDataset(train["text"], y_train, tok, args.max_length)
    ds_inner = CaptionDataset(inner["text"], inner["viral"], tok, args.max_length)
    ds_val = CaptionDataset(val["text"], None, tok, args.max_length)

    steps_per_epoch = int(np.ceil(len(ds_train) / args.batch_size))
    eval_steps = max(1, steps_per_epoch // 4)  # evaluate on the inner hold-out 4x per epoch
    # 6% linear warm-up, as a step count (works on transformers 4.x and 5.x)
    warmup_steps = int(0.06 * steps_per_epoch * args.epochs)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=512,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,       # best checkpoint by INNER hold-out PR-AUC
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_inner,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60
    inner_metrics = trainer.evaluate()

    prob = positive_prob(trainer.predict(ds_val).predictions)
    tag = "_shuffled" if args.shuffle_labels else ""
    pd.DataFrame({"video_id": val["video_id"].to_numpy(), "prob": prob}).to_parquet(
        out / f"val_predictions_distilbert{tag}.parquet", index=False)
    info = {"args": vars(args), "train_rows": len(train), "inner_rows": len(inner),
            "val_rows": len(val), "train_minutes": round(train_minutes, 1),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "inner_pr_auc_best": inner_metrics.get("eval_pr_auc"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (out / f"run_info{tag}.json").write_text(json.dumps(info, indent=2))
    if not args.shuffle_labels:
        trainer.save_model(str(out / "model"))
        tok.save_pretrained(str(out / "model"))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
