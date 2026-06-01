# 机构：人工智能研究所
# 人员：东
# 时间：2026/5/31 18:56

import torch

# =========================路径配置==========================
DATA_ROOT = "./data"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# =========================数据集配置==========================
DATASET_NAME = "CIFAR10"
NUM_CLASSES = 10
IN_CHANNELS = 3
IMAGE_SIZE = 32
VAL_RATIO = 0.1
RANDOM_SEED = 42

# =========================模型配置==========================
DROPOUT = 0.5

# =========================训练配置==========================
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# =========================设备配置==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================文件保存==========================
BEST_MODEL_PATH = CHECKPOINT_DIR + "/alexnet_cifar10_best.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR + "/alexnet_cifar10_last.pth"

TRAIN_LOG_PATH = LOG_DIR + "/train_log.csv"
LOSS_CURVE_PATH = LOG_DIR + "/loss_curve.png"
ACC_CURVE_PATH = LOG_DIR + "/acc_curve.png"

