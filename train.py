# 机构：人工智能研究所
# 人员：东
# 时间：2026/5/31 18:58
import csv

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms

import config

from models.alexnet import AlexNet

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (correct == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    return avg_val_loss, val_acc


def main():

    # =================设备===================
    device = config.DEVICE

    print("Using Device:", device)

    # ================数据预处理=================
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # ===================完整训练集====================
    full_train_dataset = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train
    )

    # ===================划分 train / val==================
    val_size = int(len(full_train_dataset) * config.VAL_RATIO)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator = torch.Generator().manual_seed(
            config.RANDOM_SEED
        )
    )

    # ===================DataLoader==================
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # =====================模型=========================
    model = AlexNet().to(device)

    # ===================损失函数=======================
    criterion = nn.CrossEntropyLoss()

    # ====================优化器======================
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    best_val_acc = 0.0
    # ====================开始训练====================
    for epoch in range(config.EPOCHS):
        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Acc: {epoch_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}"
        )

        with open(config.TRAIN_LOG_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                epoch_loss,
                epoch_acc,
                val_loss,
                val_acc
            ])


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, config.BEST_MODEL_PATH)

    # =========================保存模型=============================
    torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, config.LAST_MODEL_PATH)
    print("Model Saved!")


if __name__ == "__main__":
    main()


















