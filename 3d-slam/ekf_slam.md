# EKF SLAM

## Overview

EKF SLAM 指的是使用扩展卡尔曼滤波器来同时估计:

* 机器人 / 相机自身状态
* 地图中路标点状态

它的核心思想是:

1. 用系统运动模型做预测
2. 用观测模型做校正
3. 把位姿和地图统一放进同一个高斯状态里

---

## 状态表示

最经典的 EKF SLAM 状态向量可以写成:

$$
\mathbf{x}=
\begin{bmatrix}
\mathbf{x}_r\\
\mathbf{m}_1\\
\mathbf{m}_2\\
\cdots\\
\mathbf{m}_N
\end{bmatrix}
$$

其中:

* $\mathbf{x}_r$: 机器人或相机状态
* $\mathbf{m}_i$: 第 $i$ 个路标点

同时维护协方差矩阵:

$$
P
$$

它不仅表示每个变量自己的不确定性, 也表示它们之间的相关性。

---

## 预测步骤

给定控制输入 $\mathbf{u}_k$, 运动模型写成:

$$
\mathbf{x}_{k|k-1}=f(\mathbf{x}_{k-1},\mathbf{u}_k)
$$

协方差更新为:

$$
P_{k|k-1}=F_kP_{k-1}F_k^T+Q_k
$$

其中:

* $F_k$: 运动模型对状态的雅可比
* $Q_k$: 过程噪声

---

## 更新步骤

若第 $k$ 时刻观测为 $\mathbf{z}_k$, 观测模型写成:

$$
\mathbf{z}_k=h(\mathbf{x}_k)+\mathbf{v}_k
$$

线性化后有:

$$
H_k=\frac{\partial h}{\partial \mathbf{x}}
$$

创新为:

$$
\mathbf{y}_k=\mathbf{z}_k-h(\mathbf{x}_{k|k-1})
$$

创新协方差:

$$
S_k=H_kP_{k|k-1}H_k^T+R_k
$$

卡尔曼增益:

$$
K_k=P_{k|k-1}H_k^TS_k^{-1}
$$

状态更新:

$$
\mathbf{x}_{k}=\mathbf{x}_{k|k-1}+K_k\mathbf{y}_k
$$

协方差更新:

$$
P_k=(I-K_kH_k)P_{k|k-1}
$$

---

## 特点

EKF SLAM 的优点是:

* 概率意义清晰
* 预测与更新步骤明确
* 对小规模问题有效

它的主要缺点是:

* 线性化误差会积累
* 状态维度增大后协方差矩阵成本很高
* 地图点太多时可扩展性较差
