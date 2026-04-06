# SE(3)

## Overview

SE(3) 表示三维空间中的刚体变换群。  
一个 SE(3) 变换只包含:

* 旋转
* 平移

不包含尺度变化。

标准形式为:

$$
T(x)=Rx+t,\qquad R\in SO(3),\ t\in\mathbb{R}^3
$$

齐次坐标形式可写成:

$$
T=
\begin{bmatrix}
R & t\\
0^\top & 1
\end{bmatrix}
\in \mathrm{SE}(3)
$$

---

## 自由度

SE(3) 一共有 6 个自由度:

* 3 个旋转自由度
* 3 个平移自由度

因此在优化里常把位姿增量写成:

$$
\xi \in \mathbb{R}^6
$$

再通过指数映射更新到群上。

---

## 几何含义

SE(3) 适用于:

* 同一刚体在不同坐标系中的表示
* 两帧之间的相机位姿关系
* 尺度已经统一的点云 / 地图对齐

它的核心假设是:

* 场景满足刚体关系
* 不存在全局尺度漂移

---

## 在点集对齐中的形式

若两组对应点满足刚体关系, 则常见目标写成:

$$
\min_{R\in SO(3),\,t\in\mathbb{R}^3}
\sum_{i=1}^{N}
\|p_i-(Rq_i+t)\|^2
$$

如果对应关系已知, 可以直接使用 SVD 闭式求解。  
如果对应关系未知, 则常和 ICP 联合使用。

---

## 与其他群的区别

### 与 SO(3) 的区别

SO(3) 只表示旋转:

$$
x' = Rx
$$

SE(3) 则进一步加入平移:

$$
x' = Rx+t
$$

### 与 Sim(3) 的区别

Sim(3) 还允许全局尺度:

$$
x' = sRx+t
$$

因此单目重建、局部重建拼接、尺度漂移场景里, 往往不能只用 SE(3)。

---

## 常见相关页面

* [Point Alignment](point_alignment.md)
* [ICP](icp.md)
* [SE(3) Point Alignment](se3_point_alignment.md)
* [Lie Group and Lie Algebra](../optimization/lie_group_and_lie_algebra.md)
