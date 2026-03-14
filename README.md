---
description: NeRF
---

# NeRF

## NeRF

NeRF 要学的是一个连续场景函数：

$$
F_\theta:(\mathbf{x},\mathbf{d}) \longrightarrow (\mathbf{c},\sigma)
$$

其中：

* $\mathbf{x}\in\mathbb{R}^3$：空间中的 3D 点
* $\mathbf{d}\in\mathbb{R}^3$：观察方向
* $\mathbf{c}\in\mathbb{R}^3$：该点在该方向下的颜色
* $\sigma\in\mathbb{R}$：体密度，表示该点“有多不透明”

最终目标是：

$$
(\text{camera pose},\ K,\ H,\ W) \;\longrightarrow\; I \in \mathbb{R}^{H\times W\times 3}
$$

也就是：

* 输入：相机位姿（外参）+ 内参 + 分辨率
* 输出：一张图片

### 1. 从像素出发生成射线

对于图像上的每个像素 $(x,y)$，NeRF 先构造一条从相机中心出发的射线：

$$
\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}
$$

其中：

* $\mathbf{o}$：射线起点，即相机中心
* $\mathbf{d}$：射线方向
* $t$：沿射线的深度

先在相机坐标系下把像素反投影成方向：

$$
\mathbf{d}_{cam}=
\begin{bmatrix}
(x-c_x)/f_x \\
-(y-c_y)/f_y \\
-1
\end{bmatrix}
$$

再用相机外参中的旋转矩阵把它变到世界坐标系：

$$
\mathbf{d}_{world}=R\mathbf{d}_{cam}
$$

相机中心直接由 `c2w` 的平移项给出：

$$
\mathbf{o}= \mathbf{t}
$$

对应到当前仓库，render\_preprocess.py 里已经实现了：

* `compute_pixel_direction`：单个像素生成世界坐标下的射线方向
* `get_rays`：整张图像生成 `rays_o` 和 `rays_d`
* `sample_random_rays_od`：随机采样部分射线用于训练
* `sample_rays_color_by_coords`：取出这些射线对应的 GT 像素颜色

### 2. 为什么训练时只采样部分像素

如果每次都对整张图的所有像素做体渲染，计算量会非常大。\
所以训练时通常只随机采样 $N\_{rand}$ 条射线。

这样做的好处：

* 降低显存和计算开销
* 训练更像 SGD / mini-batch
* 多轮迭代后仍能覆盖整张图像

### 3. 沿射线采样 3D 点

对每条射线，不是直接预测整条射线的颜色，而是先在射线上取若干个点：

$$
\mathbf{x}_i=\mathbf{o}+t_i\mathbf{d}
$$

其中 $t\_i$ 是 near 到 far 之间的一组采样深度。

常见做法：

* 均匀采样
* 分层采样（stratified sampling）
* coarse-to-fine 的层次采样

这些采样点会送进 MLP，预测每个点的颜色和密度。

### 4. MLP 学什么

NeRF 的 MLP 输入是一个空间点和方向，输出颜色与密度：

$$
(\mathbf{x}_i,\mathbf{d}) \xrightarrow{\text{MLP}} (\mathbf{c}_i,\sigma_i)
$$

这里的直觉是：

* 密度 $\sigma\_i$ 决定该位置是否“有物体”
* 颜色 $\mathbf{c}\_i$ 决定如果看到这个位置，它应该贡献什么颜色

通常会先对输入做 positional encoding，增强高频表达能力：

$$
\gamma(p)=\left[\sin(2^0\pi p),\cos(2^0\pi p),\dots,\sin(2^{L-1}\pi p),\cos(2^{L-1}\pi p)\right]
$$

原因是：普通 MLP 更偏向低频函数，不容易直接表示复杂细节。

### 5. 体渲染：把一串点合成一个像素颜色

沿射线每个采样点都会得到 $(\mathbf{c}\_i,\sigma\_i)$，接下来使用体渲染公式做加权累积。

先定义第 $i$ 个采样点的 alpha：

$$
\alpha_i = 1-\exp(-\sigma_i \delta_i)
$$

其中 $\delta\_i=t\_{i+1}-t\_i$，表示该采样区间长度。

再定义透射率：

$$
T_i=\prod_{j=1}^{i-1}(1-\alpha_j)
$$

它表示光线到达第 $i$ 个点之前没有被前面挡住的概率。

于是第 $i$ 个点对最终像素颜色的权重是：

$$
w_i=T_i\alpha_i
$$

最终渲染颜色：

$$
\hat{\mathbf{C}}(\mathbf{r})=\sum_i w_i\mathbf{c}_i
$$

这一步是 NeRF 的核心：\
不是在 2D 图像上直接预测像素，而是先建模 3D 场，再通过可微体渲染得到像素。

### 6. 训练目标

训练时我们有真实图像中的像素颜色 $\mathbf{C}(\mathbf{r})$，也有渲染得到的预测颜色 $\hat{\mathbf{C\}}(\mathbf{r})$。

最常见的损失就是 MSE：

$$
\mathcal{L}=\sum_{\mathbf{r}\in\mathcal{R}} \left\| \hat{\mathbf{C}}(\mathbf{r})-\mathbf{C}(\mathbf{r}) \right\|_2^2
$$

也就是：

* 从训练图像中随机取一些像素
* 根据相机参数生成这些像素对应的射线
* 沿射线采样点，经过 MLP 和体渲染得到预测颜色
* 与真实像素颜色做误差，反向传播更新参数

### 7. 整个 NeRF 流程可以概括成

1. 输入相机内外参和图像
2. 为像素生成射线
3. 随机采样一批射线
4. 沿每条射线采样多个 3D 点
5. MLP 预测每个点的颜色和密度
6. 用体渲染把点的贡献合成为像素颜色
7. 与真实像素做监督，更新网络

### 8. 这个仓库当前做到哪

当前实现主要完成了“渲染前处理”这一部分：

* 已完成：像素坐标 $\rightarrow$ 射线起点/方向
* 已完成：从整张图中随机采样训练射线
* 已完成：提取这些射线对应的真实颜色
* 未完成：位置编码、MLP、体渲染、训练循环、评估

所以目前可以把项目理解成：

> 已经搭好了 NeRF 输入管线的前半段，但还没有进入“网络预测 + 可微渲染 + 优化”。

### 9. 一句话理解 NeRF

NeRF 本质上是在学习一个 3D 连续辐射场：\
给定空间位置和观察方向，网络告诉你这里有多少“东西”（密度）以及它看起来是什么颜色，然后通过体渲染把整条射线上的信息合成为最终图像。
