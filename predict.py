import argparse
from collections import defaultdict
import os
import logging

from datasets import Dataset
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from api import tinderAPI
from image_classifier import ResNet, load_images
from helpers import try_gpu
from text_classifier import prep_data


def make_prediction(
    text_preds: np.ndarray,
    image_preds: np.ndarray,
    text_weight: float,
    image_weight: float,
    threshold: float = 0.5
):
    total_pred = text_preds * text_weight + image_preds * image_weight
    return np.where(total_pred > threshold, True, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-token",
        type=str,
        required=True
    )
    parser.add_argument(
        "--image-model-path",
        type=str,
        default="models/image_model.pt"
    )
    parser.add_argument(
        "--text-model-path",
        type=str,
        default="models/text_model.pt"
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Valid huggingface tokenizer name"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1
    )
    parser.add_argument(
        "--image-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--bio-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images/pred"
    )
    parser.add_argument(
        "--save-preds-path",
        type=str,
        default="preds.txt"
    )
    args = parser.parse_args()
    assert args.image_weight + args.bio_weight == 1, "Prediction weights must sum up to 1"

    logger = logging.getLogger(__name__)
    logger.setLevel(20)  # 20 = INFO
    logging.basicConfig(format="%(asctime)s : %(levelname)s | %(message)s")

    device = try_gpu()
    logger.info(f"Inference is done on {device} for {args.num_requests} requests")

    # Init models
    # TODO: Is there a better way to init the model object with non hard-coded args?
    image_model = ResNet(3, 64, [2, 2, 2, 2], num_classes=2)
    image_model.load_state_dict(torch.load(args.image_model_path))
    image_model.to(device)

    # As we didn't do any changes to the tokenizer, just load it from HF
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model_path)
    text_model.to(device)

    # We'll predict with Trainer, because it's the fastest
    trainer = Trainer(model=text_model)

    # Get profiles
    api = tinderAPI(args.api_token)

    # Add sub folder to the images folder so torchvision.ImageDataset works with ease...
    unc_images_path = os.path.join(args.image_dir, "unc")
    os.makedirs(unc_images_path, exist_ok=True)
    profile_info = defaultdict(list)
    for _ in range(args.num_requests):
        profiles = api.get_profiles()
        for profile in profiles:
            profile_info["id"].append(profile.id)
            profile_info["bio"].append(profile.bio)
            profile.download_photos(unc_images_path)

    # Prep data and predict
    # Text
    text_dataset = Dataset.from_dict(profile_info)
    text_dataset = prep_data(text_dataset, tokenizer, cols_to_drop=["id"])
    text_preds = np.max(trainer.predict(text_dataset).predictions, axis=-1)

    # Images
    images_dataset = load_images(args.image_dir, 8, shuffle=False, transformations=[torchvision.transforms.ToTensor()])
    image_preds = []
    image_model.eval()
    with torch.no_grad():
        for X, _ in images_dataset:
            image_preds.append(F.softmax(image_model(X.to(device)).detach().numpy()))
    image_preds = np.max(np.concatenate(image_preds), axis=-1)

    # Total predictions
    preds = make_prediction(text_preds, args.bios_weight, image_preds, args.image_weight)

    # Preds are ordered by id, so we like them if the entry is true in preds
    liked = 0
    for user_id, to_like in zip(profile_info, preds):
        if to_like:
            api.like(user_id)
            liked += 1

    logger.info(f"Liked: {liked} profiles, disliked: {len(preds) - liked}")

    if args.save_preds_path is not None:
        os.makedirs(args.save_preds_path, exist_ok=True)
        with open(args.save_preds_path, "wt") as f:
            f.writelines("\n".join([f"{user_id}: {to_like}" for user_id, to_like in zip(profile_info, preds)]))

        logger.info(f"Wrote results to {args.save_preds_path}")

    logger.info("All done")
