# 1. AlexNet 论文梳理

《ImageNet Classification with Deep Convolutional Neural Networks》

作者：Alex Krizhevsky 等人，发表于：NIPS 2012

## 1.1 数据集ImageNet LSVRC 2010/2012

* 超大规模数据集
* 约 120 万训练图像
* 1000 类分类任务
* RGB 彩色图像

这是第一次深层 CNN 在超大数据集上取得巨大成功，之前 CNN 主要还是 MNIST 小数据集

## 1.2 网络结构

```text
CNN → Pool
CNN → Pool
CNN
CNN
CNN → Pool
FC
FC
FC
```

---

## 1.3 重要创新

### 1.3.1 ReLU 激活函数

Sigmoid / Tanh 梯度消失、收敛慢；ReLU 训练速度大幅提升，深层网络首次真正可训练。

$
ReLU(x)=max(0,x)
$

### 1.3.2 GPU 训练

双 GPU 并行训练，显著加速训练过程。

### 1.3.3 局部响应归一化（LRN）

论文里用了Local Response Normalization，增强局部竞争。BatchNorm 出现后基本淘汰，而且在 VGG 里证明此方法效果不大。

### 1.3.4 重叠池化（Overlapping Pooling）

以前池化窗与步长相等 kernel=stride=2，文章里使用 kernel=3 stride=2，池化区域重叠，以此保留更多信息。

### 1.3.5 减少过拟合

（1）数据增强

* 平移
* 翻转
* 随机裁剪
* PCA颜色扰动

（2）Dropout最大工程创新之一

训练时随机丢弃神经元，防止神经元过度依赖。

## 1.4 训练细节

Batch Size= 128、SGD + Momentum

## 1.5 评价指标

核心含义：假设我们拿一张“猫”的图片让网络去猜，网络最终会输出所有类别的概率得分。我们将这些得分从高到低进行排序。

1. Top-1 准确率（Top-1 Accuracy）含义：模型预测的概率最高的那一个类别（即排在第 1 名的类别），是否等于真实标签。

   示例：模型输出：[猫: 40%, 狗: 35%, 猪: 15%, 兔: 10%]，第 1 名是“猫”，刚好和真实标签一致 → Top-1 预测正确。如果第 1 名是“狗”，哪怕“猫”排在第 2 名（概率 39.9%），在 Top-1 的标准下也算预测错误。

2. Top-5 准确率（Top-5 Accuracy）含义：模型预测的概率最高的前 5 个类别里，是否包含了真实标签。只要包含了，就算猜对。

   示例：模型输出：[狗: 40%, 狼: 25%, 豹: 15%, 猫: 12%, 虎: 8%]，虽然模型最看好“狗”（Top-1 猜错了），但是真实标签“猫”挤进了前 5 名（排在第 4） → Top-5 预测正确。

