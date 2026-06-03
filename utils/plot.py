# 机构：人工智能研究所
# 人员：东
# 时间：2026/5/31 19:02
import os

import numpy as np
import torch

# 设置环境变量，允许重复加载库，避免因库冲突导致的错误。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import matplotlib.pyplot as plt
import config

plt.rcParams["font.family"] = "Times New Roman"

def plot_train_curves():
    log_data = pd.read_csv(config.TRAIN_LOG_PATH, header=None)

    log_data.columns = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc"
    ]

    # ===========================找实验分界==============================
    experiments = []
    start_ix = 0
    for i in range(1, len(log_data)):
        if log_data.loc[i, "epoch"] == 1:
            experiments.append(log_data.iloc[start_ix:i])
            start_ix = i

    experiments.append(log_data.iloc[start_ix:])

    colors = ['r', 'g', 'b', 'c', 'm']
    labels = ['BaseLine', 'NoDropout', 'Tanh', 'NoDataAug', 'NoWeightDecay']

    # ============================Loss曲线=============================
    plt.figure(figsize=(10,5))
    plt.rcParams["font.size"] = 14
    for idx, exp in enumerate(experiments):
        plt.plot(exp["epoch"], exp["train_loss"], color=colors[idx], label=f"{labels[idx]}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.LOSS_CURVE_PATH)
    plt.close()

    # ===========================Accuracy曲线==============================
    plt.figure(figsize=(10, 5))
    plt.rcParams["font.size"] = 14
    for idx, exp in enumerate(experiments):
        plt.plot(exp["epoch"], exp["val_acc"], color=colors[idx], label=f"{labels[idx]}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.ACC_CURVE_PATH)
    plt.close()


def visualize_predictions(model, dataset, device):
    classes = dataset.classes
    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.2023, 0.1994, 0.2010]
    ).view(3, 1, 1)

    indices = torch.randperm(len(dataset))[:16]

    plt.figure(figsize=(12, 12))
    plt.rcParams["font.size"] = 20
    model.eval()

    with torch.no_grad():

        for i, idx in enumerate(indices):

            image, label = dataset[idx]

            output = model(
                image.unsqueeze(0).to(device)
            )

            pred = output.argmax(1).item()

            image = image.cpu()

            image = image * std + mean

            image = torch.clamp(image, 0, 1)

            image = np.transpose(
                image.numpy(),
                (1, 2, 0)
            )

            plt.subplot(4, 4, i + 1)

            plt.imshow(image, interpolation="bicubic")

            plt.title(
                f"P:{classes[pred]}\nT:{classes[label]}",
                fontsize=8
            )

            plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "./logs/prediction_examples.png",
        dpi=300
    )
    plt.close()

if __name__ == "__main__":
    plot_train_curves()

