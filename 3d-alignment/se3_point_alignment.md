# SE(3) Point Alignment

## Overview

如果两组 3D 点已经有一一对应关系, 则刚体配准可以直接闭式求解, 不需要 ICP 那样反复猜最近邻。

问题写成:

$$
\min_{R\in SO(3),\,t\in\mathbb{R}^3}
\sum_{i=1}^{N}
\|p_i-(Rq_i+t)\|^2
$$

这类解法常被称为:

* Arun's Method
* Kabsch Algorithm
* SVD closed-form solution

---

## 步骤

### 1. 计算质心

$$
\bar p=\frac{1}{N}\sum_i p_i,\qquad
\bar q=\frac{1}{N}\sum_i q_i
$$

### 2. 去中心化

$$
p_i'=p_i-\bar p,\qquad
q_i'=q_i-\bar q
$$

### 3. 构造协方差矩阵

$$
H=\sum_{i=1}^{N}p_i'{q_i'}^T
$$

### 4. 对 H 做 SVD

$$
H=U\Sigma V^T
$$

### 5. 求旋转

$$
R=USV^T
$$

其中 $S$ 用来保证:

$$
\det(R)=1
$$

### 6. 求平移

$$
t=\bar p-R\bar q
$$

---

## 与 ICP 的关系

它和 ICP 的关系是:

* 对应关系已知时: 直接用 SVD 闭式求解
* 对应关系未知时: 需要 ICP 先建立对应, 再反复调用这个闭式解

因此可以把 point-to-point ICP 理解成:

* 外层: 对应关系迭代
* 内层: SE(3) 闭式配准

---

## 适用条件

SE(3) 对齐适用于:

* 两组点尺度一致
* 场景满足刚体关系
* 已知一一对应关系

如果存在尺度漂移, 就需要转向 Sim(3) 对齐。
