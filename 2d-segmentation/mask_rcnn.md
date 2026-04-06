# Mask R-CNN

## Overview

Mask R-CNN 是在 Faster R-CNN 基础上扩展出来的实例分割模型。  
它不仅预测:

* 目标类别
* 边界框

还会额外预测:

* 像素级实例掩码

---

## 从 R-CNN 到 Fast / Faster R-CNN

### 1. R-CNN

R-CNN 的基本思路是:

1. 先提出很多候选区域
2. 对每个候选区域分别裁剪
3. 再送入 CNN 做分类

优点是检测思路清晰。  
缺点是:

* 区域数目多
* 每个候选框都要单独跑 CNN
* 速度很慢

### 2. Fast / Faster R-CNN

Fast R-CNN 把卷积特征提取共享起来。  
Faster R-CNN 进一步引入 RPN, 用网络直接生成 proposal。

于是两阶段检测流程变成:

1. backbone 提取整图特征
2. RPN 产生 proposals
3. ROI 特征送入分类头和边框回归头

---

## Mask R-CNN 的核心改进

Mask R-CNN 在 Faster R-CNN 的基础上增加了一个 mask branch。

因此输出变成三部分:

1. 分类
2. 边界框回归
3. 掩码预测

其多任务损失通常写成:

$$
\mathcal{L}=
\mathcal{L}_{cls}
+\mathcal{L}_{box}
+\mathcal{L}_{mask}
$$

---

## ROI Align

Mask R-CNN 最关键的工程改进之一是 **ROI Align**。

原因是:

* 原来的 ROI Pooling 会做量化
* 量化会破坏像素级对齐
* 对实例分割尤其不利

ROI Align 通过双线性插值避免粗糙取整, 从而保留更精确的空间对应关系。

因此它特别适合:

* 实例掩码预测
* 关键点定位
* 细粒度像素级任务

---

## 总结

Mask R-CNN 可以看成:

* Faster R-CNN 的检测框架
* 加上 ROI Align
* 再加一个并行 mask 分支

因此它是经典的两阶段实例分割模型。
