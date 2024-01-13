import argparse
from collections import defaultdict
import logging
import pickle
import re

from datasets import load_metric, Dataset
import emoji
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments


def clean_bios(dataset, bio_col: str):
    """Remove flags, emojis and special symbols from profiles"""
    def _cleaner(text):
        flags = re.findall(u"[\U0001F1E6-\U0001F1FF]", text)

        text = text.replace("\n", " ")
        text = "".join(w for w in text if w not in emoji.EMOJI_DATA and w not in flags)
        text = text.replace("\u200d", "")

        return text

    return {f"{bio_col}_clean": _cleaner(dataset[bio_col])}


def tokenize(dataset, tokenizer, text_col: str, max_length: int = 512):
    return tokenizer(dataset[text_col], padding="max_length", truncation=True, max_length=max_length)


def prep_data(dataset, tokenizer, bio_col: str = "bio", cols_to_drop: list = None):
    cols_to_drop = ["id", "name", "bio", "age"] if cols_to_drop is None else cols_to_drop
    dataset_prep = (
        dataset
        .map(clean_bios, fn_kwargs={"bio_col": bio_col}, batched=False)
        .map(tokenize, fn_kwargs={"tokenizer": tokenizer, "text_col": f"{bio_col}_clean"})
        .rename_column(f"{bio_col}_clean", "text")
        .remove_columns(cols_to_drop)
    )
    return dataset_prep


def compute_metrics(eval_pred, metric=None):
    metric = load_metric("accuracy") if metric is None else metric
    logits, labels = eval_pred
    predictions = np.array([np.argmax(logits, axis=-1)]).flatten()
    return metric.compute(references=labels, prediction_scores=predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data"
    )
    parser.add_argument(
        "--distilbert-name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Valid huggingface DistilBert name"
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Valid huggingface tokenizer name"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-10,
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=2_500,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12_500,
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=4
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    logger = logging.getLogger(__name__)
    logger.setLevel(20)  # 20 = INFO
    logging.basicConfig(format="%(asctime)s : %(levelname)s | %(message)s")

    logger.info(f"Using {data_dir}")

    # Load data
    with open(f"{data_dir}/bios.pkl", "rb") as f:
        bios = pickle.load(f)

    with open(f"{data_dir}/labels.pkl", "rb") as f:
        labels = pickle.load(f)

    logger.info(f"{len(bios)} profiles loaded")
    dataset_dict = defaultdict(list)
    unlabeled = []

    for k, v in bios.items():
        if k in labels:
            dataset_dict["id"].append(k)
            dataset_dict["label"].append(labels[k])
            for k_, v_ in v.items():
                dataset_dict[k_].append(v_)
        else:
            unlabeled.append(k)

    logger.info(f"There are {len(unlabeled)} unlabeled profiles")
    logger.info(f"Load and prep dataset with {args.test_size} test size")

    bios_dataset = Dataset.from_dict(dataset_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    bios_dataset = prep_data(bios_dataset, tokenizer)
    bios_dataset = bios_dataset.train_test_split(test_size=args.test_size)

    logger.info(f"Load {args.distilbert_name} model")
    # Init model and freeze first layers
    bert = AutoModelForSequenceClassification.from_pretrained(
        args.distilbert_name,
        num_labels=args.num_classes,
        torch_dtype=torch.float32
    )

    layers_to_train = [
        "distilbert.transformer.layer.5.ffn.lin1.weight",
        "distilbert.transformer.layer.5.ffn.lin1.bias",
        "distilbert.transformer.layer.5.ffn.lin2.weight",
        "distilbert.transformer.layer.5.ffn.lin2.bias",
        "distilbert.transformer.layer.5.output_layer_norm.weight",
        "distilbert.transformer.layer.5.output_layer_norm.bias",
        "pre_classifier.weight",
        "pre_classifier.bias",
        "classifier.weight",
        "classifier.bias",
    ]
    for n, l in bert.named_parameters():
        if n not in layers_to_train:
            l.requires_grad = False

    training_args = TrainingArguments(
        output_dir=f"models/text_model.pt",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        learning_rate=args.lr,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.device_batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="constant_with_warmup",
        save_steps=args.eval_steps,
        max_steps=args.max_steps,
    )

    # Create training pipeline
    trainer = Trainer(
        model=bert,
        args=training_args,
        train_dataset=bios_dataset["train"],
        eval_dataset=bios_dataset["test"],
        compute_metrics=compute_metrics
    )
    trainer.train()

    logger.info("All done")
