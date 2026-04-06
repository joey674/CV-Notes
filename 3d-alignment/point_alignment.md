# Point Alignment

## Overview

Point Alignment 讨论的是:

* 给定两组 2D / 3D 点
* 求一个变换
* 让两组点在同一坐标系下尽量对齐

如果对应关系已知, 问题通常写成:

$$
\min_{\mathcal{T}}
\sum_{i=1}^{N}
\|p_i-\mathcal{T}(q_i)\|^2
$$

其中:

* $\{p_i\}$: 目标点集
* $\{q_i\}$: 源点集
* $\mathcal{T}$: 待估计的几何变换

---

## 常见变换模型

Point Alignment 的核心区别, 往往不在“是否对齐”, 而在“允许什么变换”。

### 1. 刚体变换

如果只允许旋转和平移, 则:

$$
\mathcal{T}(x)=Rx+t,\qquad R\in SO(3)
$$

这对应 **SE(3) Point Alignment**。

### 2. 相似变换

如果还允许全局尺度变化, 则:

$$
\mathcal{T}(x)=sRx+t
$$

这对应 **Sim(3) Point Alignment**。

### 3. 更一般的射影 / 单应变换

如果场景存在更复杂的形变, 可能需要:

* 仿射变换
* 单应变换
* 射影变换
* SL(4) / projective alignment

---

## 两类问题

### 1. 对应关系已知

如果每个 $q_i$ 对应哪个 $p_i$ 已知, 问题通常可以闭式求解:

* SE(3): 常用 SVD / Kabsch / Arun 方法
* Sim(3): 常用 Umeyama 方法

### 2. 对应关系未知

如果对应关系未知, 则需要交替进行:

1. 建立对应
2. 在当前对应下求变换

这类代表方法是 ICP。

---

## 常见拆分方式

从笔记组织角度看, Point Alignment 可以继续拆成:

* 问题形式: Point Alignment
* 变换模型: [SE(3)](se3.md)、[Sim(3)](sim3.md)
* 具体算法: [ICP](icp.md)、[SE(3) Point Alignment](se3_point_alignment.md)、[Sim(3) Umeyama](umeyama.md)
* 鲁棒扩展: [IRLS + Umeyama](irls_umeyama.md)、[SL(4) Point Alignment](sl4_alignment.md)

---

## 在重建系统中的作用

Point Alignment 常出现在:

* 点云配准
* 局部地图拼接
* 长序列 chunk 对齐
* SLAM / SfM / 前馈几何模型的坐标系对齐

因此它既是一个基础几何问题, 也是很多系统级方法的中间步骤。
