import argparse
import logging
import os
import pickle

import numpy as np
from tqdm.auto import tqdm

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from helpers import Plotter, try_gpu


def load_images(
    data_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 1,
    transformations: list = None,
    train_size: float = None,
    seed: int = 42
):
    transforms = torchvision.transforms.Compose(transformations) if transformations else None
    data = torchvision.datasets.ImageFolder(data_path, transforms)

    if train_size is not None:
        _train_size = int(len(data) * train_size)
        _val_size = len(data) - _train_size

        print("Train dataset size:", _train_size, "test dataset size:", _val_size)

        gen = torch.Generator().manual_seed(seed)
        data_train, data_val = random_split(data, [_train_size, _val_size], generator=gen)

        return (
            DataLoader(data_train, batch_size, shuffle, num_workers=num_workers),
            DataLoader(data_val, batch_size, shuffle, num_workers=num_workers)
        )

    return DataLoader(data, batch_size, shuffle, num_workers=num_workers)


def init_weights_(layer):
    if isinstance(layer, torch.nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode="fan_out")
    elif isinstance(layer, (nn.BatchNorm2d)):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.res_con = None
        if stride != 1 and in_channels != out_channels:
            self.res_con = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, X):
        input_ = X
        out = self.bn_1(self.conv_1(X))
        out = self.relu(out)
        out = self.bn_2(self.conv_2(out))

        if self.res_con is not None:
            out += self.res_con(input_)

        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channels, out_channels, layers, num_classes=10):
        self.hidden_size = out_channels

        super().__init__()
        self.conv_1 = nn.Conv2d(
            input_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.res_1 = self.make_res_block(out_channels, out_channels, layers[0], 1)
        self.res_2 = self.make_res_block(out_channels, 128, layers[1], 2)
        self.res_3 = self.make_res_block(128, 256, layers[2], 2)
        self.res_4 = self.make_res_block(256, 512, layers[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.out = nn.Linear(512, num_classes, bias=False)

    def make_res_block(self, input_channels, output_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.hidden_size, output_channels, stride))
            self.hidden_size = output_channels

        return nn.Sequential(*layers)

    def forward(self, X):
        out = self.bn_1(self.conv_1(X))
        out = self.relu(out)

        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = self.res_4(out)

        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.out(out)

        return out


def train_model(
    net, train_iter, test_iter, epochs, optim, loss=None, device=None,
    init_weights=False, debug=False, save_model=None,
    verbose_interval=5, scheduler=None, clip_grad=False
):
    # Init stuff
    if init_weights:
        net.apply(init_weights_)

    loss = torch.nn.CrossEntropyLoss() if not loss else loss
    plotter = Plotter(
        x_label="epochs",
        y_label="loss",
        x_lim=[1, epochs],
        legend=["train loss", "test loss"]
    )

    num_batches = len(train_iter)

    device = try_gpu() if not device else device
    print(f"Training on {device}")
    net.to(device)

    # Training loop
    for epoch in range(epochs):
        net.train()
        train_loss_ = []

        for i, (X, y) in enumerate(tqdm(train_iter)):
            optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_pred = net(X)
            l = loss(y_pred, y)
            l.backward()
            train_loss_.append(l.item())

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2., norm_type=2)
            optim.step()

            if debug:
                break

        with torch.no_grad():
            net.eval()

            test_loss_ = []
            for X_test, y_test in test_iter:
                X_test, y_test = X_test.to(device), y_test.to(device)
                pred_test = net(X_test)
                test_loss_.append(loss(pred_test, y_test).item())

            train_loss = np.mean(train_loss_)
            test_loss = np.mean(test_loss_)

            plotter.add(epoch + 1, (train_loss, test_loss))

        if (epoch + 1) % verbose_interval == 0 or epoch == 0 or epoch == (epochs - 1):
            plotter.plot()
            print(
                f"epoch: {epoch}", f'train loss: {train_loss:.3f}, test loss: {test_loss:.3f}, '
                                   f"lr: {optim.param_groups[0]['lr']:.5f}"
            )

        if debug:
            break

        if scheduler is not None:
            scheduler.step()

    if save_model is not None:
        os.makedirs(save_model, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(save_model, "image_model.pt"))

        print(f"Optimizer saved to {save_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=False,
        type=str,
        default="data"
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        default=64
    )
    parser.add_argument(
        "--train-size",
        required=False,
        type=float,
        default=0.85
    )
    parser.add_argument(
        "--num-classes",
        required=False,
        type=int,
        default=2
    )
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default=100
    )
    parser.add_argument(
        "--lr",
        required=False,
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--weight-decay",
        required=False,
        type=float,
        default=5e-4
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    logger = logging.getLogger(__name__)
    logger.setLevel(20)  # 20 = INFO
    logging.basicConfig(format="%(asctime)s : %(levelname)s | %(message)s")

    logger.info(f"Using {data_dir}")

    logger.info("Prep image dataset")
    # ImageFolder expects images to be separated by classes
    # It's not the case for us, so we'll have to manually update them
    with open(f"{data_dir}/labels.pkl", "rb") as f:
        labels = pickle.load(f)

    no_lbl = set()
    for img_path in os.listdir(f"{data_dir}/images"):
        profile_id = img_path.split("_")[0]
        # Move labeled photos into two directories with corresponding labels
        if os.path.isfile(f"{data_dir}/images/{img_path}"):
            try:
                os.renames(f"{data_dir}/images/{img_path}", f"{data_dir}/images/{labels[profile_id]}/{img_path}")
            except KeyError:
                no_lbl.add(profile_id)

    logger.warning(f"{len(no_lbl)} profiles don't have a label yet")

    # Load images as ImageFolder dataset
    # Define the transformations that'll do to images
    transforms = [
        torchvision.transforms.Resize((128, 64)),
        torchvision.transforms.RandomGrayscale(0.3),
        torchvision.transforms.RandomCrop((96, 48), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ]
    BATCH_SIZE = args.batch_size

    train_images, val_images = load_images(
        f"{data_dir}/images",
        BATCH_SIZE,
        transformations=transforms,
        train_size=args.train_size
    )

    logger.info("Train model")
    resnet_50 = ResNet(3, 64, [2, 2, 2, 2], num_classes=args.num_classes)
    EPOCHS = args.epochs
    lr = args.lr
    WD = args.weight_decay
    optim = torch.optim.AdamW(resnet_50.parameters(), lr, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-4)

    train_model(
        resnet_50,
        train_images,
        val_images,
        EPOCHS,
        optim,
        init_weights=True,
        scheduler=scheduler,
        debug=True,
        save_model=f"models"
    )

    logger.info("All done")

