# 机构：人工智能研究所
# 人员：东
# 时间：2026/5/31 18:58

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import config

from models.alexnet import AlexNet


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

    # ===================训练集====================
    train_dataset = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train
    )

    # ===================DataLoader==================
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
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

        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}]"
            f"Loss: {epoch_loss:.4f}"
            f"Acc: {epoch_acc:.2f}%"
        )

    # =========================保存模型=============================
    torch.save(model.state_dict(), config.LAST_MODEL_PATH)
    print("Model Saved!")


if __name__ == "__main__":
    main()


















