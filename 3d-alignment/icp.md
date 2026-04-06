# ICP

## Overview

ICP 的全称是 **Iterative Closest Point**。  
它用于把两组点云做刚体配准。

标准 point-to-point ICP 的目标是求解:

$$
\min_{R\in SO(3),\,t\in\mathbb{R}^3}
\sum_{i=1}^{N}
\|Rp_i+t-q_i\|^2
$$

其中:

* $\{p_i\}$: 源点云
* $\{q_i\}$: 目标点云

---

## 为什么 ICP 要迭代

如果点对对应关系已经已知, 那么 $(R,t)$ 可以闭式求解。  
ICP 真正困难的地方在于:

* 对应关系未知

因此 ICP 每次迭代都要做两件事:

1. 建立对应
2. 在当前对应下求最优刚体变换

---

## 一次迭代的流程

给定当前估计 $(R,t)$:

### 1. 变换源点

$$
\tilde p_i=Rp_i+t
$$

### 2. 建立最近邻对应

$$
q_i=
\arg\min_{q\in Q}\|\tilde p_i-q\|
$$

### 3. 计算质心

$$
\bar p=\frac{1}{N}\sum_i \tilde p_i,\qquad
\bar q=\frac{1}{N}\sum_i q_i
$$

### 4. 去中心化

$$
p_i'=\tilde p_i-\bar p,\qquad
q_i'=q_i-\bar q
$$

### 5. 构造互协方差矩阵

$$
H=\sum_{i=1}^{N}p_i'{q_i'}^T
$$

### 6. 做 SVD

$$
H=U\Sigma V^T
$$

### 7. 求增量旋转和平移

$$
R_\Delta=VSU^T
$$

其中 $S$ 用来修正反射情形。

平移则由质心对齐得到:

$$
t_\Delta=\bar q-R_\Delta\bar p
$$

### 8. 组合到全局

$$
R\leftarrow R_\Delta R,\qquad
t\leftarrow R_\Delta t+t_\Delta
$$

---

## 特点

ICP 的特点是:

* 不需要预先知道点的匹配关系
* 局部收敛快
* 对初值敏感
* 对外点、动态物体和局部缺失比较敏感

因此它常被用作:

* 精配准
* 局部对齐
* 在已有较好初值下的 refinement

---

## point-to-point ICP 的前提

point-to-point ICP 默认假设:

* 两组点云尺度一致
* 两组点云主要满足刚体变换

因此它本质上是在做 **SE(3) / SO(3)+t** 对齐, 而不是 Sim(3) 对齐。
