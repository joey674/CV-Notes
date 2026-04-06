# DINO

## Overview

DINO 是一类视觉自监督学习方法, 核心思想是:

* 不使用人工标签
* 通过 student / teacher 自蒸馏学习稳定的表征

它不是传统意义上的对比学习“正负样本大配对”方案,  
而更像是一种不依赖标签的表征对齐方法。

---

## 基本结构

DINO 维护两个网络:

* Student
* Teacher

其中:

* student 通过反向传播更新
* teacher 不直接反向传播
* teacher 参数通过 student 参数做 EMA 更新

因此 teacher 是一个更平滑、更稳定的目标网络。

---

## 为什么不会塌缩

自蒸馏模型一个常见风险是 collapse, 也就是所有输入都映射到同一个表示。

DINO v1 通过两件事来缓解:

### 1. Centering

Centering 会让 teacher 输出更均匀, 避免所有样本集中到同一方向。

### 2. Sharpening

Sharpening 通过较低温度的 softmax 让输出重新变尖锐, 保留判别性。

因此:

* Centering: 防止塌缩
* Sharpening: 保持特征有区分度

---

## DINOv2

DINOv2 的主要特点是:

1. 更强的数据与训练规模
2. 对 centering / regularization 的进一步强化
3. 更强的特征质量与泛化能力

从路线看, DINOv2 更像是:

* 沿着 DINO 的核心范式继续做工程强化
* 而不是完全改换训练原则

---

## DINOv3

当前 PDF 中提到的一个关键词是 **Gram anchoring**。

它关注的问题是:

* 全局分类性能持续上升
* 但密集预测能力可能下降

因此 DINOv3 尝试通过 Gram anchoring 保持局部 patch 之间的空间关系, 从而减缓 dense feature collapse。

---

## 总结

DINO 的主线可以概括成:

* 用 teacher-student 自蒸馏做视觉表征学习
* 用 centering / sharpening 保持训练稳定
* 持续提升从全局语义到局部密集特征的质量
