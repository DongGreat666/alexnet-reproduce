# 机构：人工智能研究所
# 人员：东
# 时间：2026/5/31 18:58

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import config
from models.alexnet import AlexNet

def test():
    # =========================设备========================
    device = config.DEVICE
    print("Using Device:", device)

    # ========================测试集预处理=======================
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # ========================测试集========================
    test_dataset = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=False,
        download=True,
        transform=transform_test
    )

    # =======================加载器=========================
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # ======================加载模型========================
    model = AlexNet().to(device)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # =======================损失函数========================
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    # =======================测试==========================
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total

        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    test()


