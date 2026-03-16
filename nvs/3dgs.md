# 3DGS

## background
主动/被动渲染
NeRF中的ray-casting是被动渲染, 也就是说我们已知相机位姿, 从像素出发,计算每个像素点受到发光粒子影响;(这不影响NeRF是隐式辐射场的理论, 这里的意思是NeRF的前向过程是被动的,但是依旧是隐式函数表达)
3DGS则是主动渲染

---

---

## Overview

---

整体流程


---

流程总图

---





































---

## Step 1: 场景表示为一组 3D Gaussian

NeRF 用的是一个连续函数;  
而 3DGS 一开始就直接把场景表示成很多个可以被渲染的显式元素。

### 输入

通常初始化来自 SfM / COLMAP 点云:

* 初始点云坐标：`points`: $(N_g,3)$
* 对应颜色：`colors`: $(N_g,3)$

### 表示

第 $i$ 个高斯可以记成：

$$
G_i=(\mu_i,\Sigma_i,\alpha_i,\mathbf{c}_i)
$$

其中：

* `mu_i`: $(3,)$
* `Sigma_i`: $(3,3)$
* `alpha_i`: 标量
* `c_i`: $(3,)$

更常见的实现里, 协方差不会直接裸存成一个 $(3,3)$ 矩阵, 而是写成:

$$
\Sigma_i = R_i S_i S_i^T R_i^T
$$

其中：

* `R_i`: $(3,3)$，旋转
* `S_i`: $(3,3)$，通常是对角尺度矩阵

这样做的好处是:

* 更容易保证协方差矩阵是正定的
* 更方便直接优化尺度和朝向

### 输出

* 高斯中心 `mu`: $(N_g,3)$
* 高斯协方差 `Sigma`: $(N_g,3,3)$
* 高斯不透明度 `alpha`: $(N_g,1)$ 或 $(N_g,)$
* 高斯颜色/外观参数 `color` 或 `sh`: $(N_g,3)$ 或 $(N_g,C_sh)$

---

## Step 2: 世界坐标系下的高斯变换到当前相机坐标系

这一步和 NeRF 的 Step 1 有点像, 本质上也是把场景和当前相机联系起来;  
但 NeRF 是对像素发射射线, 3DGS 是把场景里的每个高斯投影到当前视角下。

### 输入

* 高斯中心 `mu`: $(N_g,3)$
* 高斯协方差 `Sigma`: $(N_g,3,3)$
* 外参 $T_w2c = [R | t]$: $(4,4)$

### 1. 中心点坐标变换

世界坐标下的高斯中心变到相机坐标系:

$$
\mu_i^{cam}=R\mu_i+t
$$

其中：

* `mu_i`: $(3,)$
* `mu_i_cam`: $(3,)$

批量后:

* `mu_cam`: $(N_g,3)$

### 2. 协方差变换

协方差在旋转下也要一起变换:

$$
\Sigma_i^{cam}=R\Sigma_iR^T
$$

这里没有平移项, 因为协方差描述的是形状和方向, 不是位置。

### 输出

* `mu_cam`: $(N_g,3)$
* `Sigma_cam`: $(N_g,3,3)$

---

## Step 3: 3D Gaussian 投影到图像平面

这一步是 3DGS 的核心之一;  
3D 空间里的一个高斯, 到图像平面上会变成一个 2D 椭圆高斯。

### 输入

* `mu_cam`: $(N_g,3)$
* `Sigma_cam`: $(N_g,3,3)$
* 内参 $K$: $(3,3)$

### 1. 中心点投影

设相机坐标系下中心点为:

$$
\mu_i^{cam}=(x_i,y_i,z_i)
$$

投影到像素平面:

$$
u_i = f_x\frac{x_i}{z_i}+c_x
$$

$$
v_i = f_y\frac{y_i}{z_i}+c_y
$$

于是高斯中心在 2D 图像中的位置是:

* `mean_2d`: $(N_g,2)$

### 2. 协方差投影

3D 协方差要通过投影函数的 Jacobian 映射到 2D：

$$
\Sigma_i^{2D}=J_i\Sigma_i^{cam}J_i^T
$$

其中 $J_i$ 是投影函数对 $(x,y,z)$ 的 Jacobian。

对于透视投影:

$$
\pi(x,y,z)=
\left(
f_x\frac{x}{z}+c_x,\ 
f_y\frac{y}{z}+c_y
\right)
$$

所以 Jacobian 大致为:

$$
J_i=
\begin{bmatrix}
f_x/z_i & 0 & -f_xx_i/z_i^2 \\
0 & f_y/z_i & -f_yy_i/z_i^2
\end{bmatrix}
$$

这样就把 $(3,3)$ 的 3D 协方差投影成 $(2,2)$ 的 2D 协方差。

### 输出

* `mean_2d`: $(N_g,2)$
* `cov_2d`: $(N_g,2,2)$
* `depth`: $(N_g,)$，通常就是每个高斯中心的 $z_i$

---

## Step 4: 可见性裁剪和 tile 划分

不是所有高斯都真的会参与每个像素的渲染;  
所以在真正 rasterization 之前, 一般会先做一层筛选和组织。

### 输入

* `mean_2d`: $(N_g,2)$
* `cov_2d`: $(N_g,2,2)$
* `depth`: $(N_g,)$
* 图像大小 $H, W$

### 1. 可见性筛选

常见会去掉这些高斯:

* 中心点在相机后方, 也就是 $z <= 0$
* 投影后完全落在图像外
* 太小或太透明, 几乎没有贡献

### 2. 估计屏幕空间覆盖范围

每个 2D 高斯都对应一个椭圆覆盖区域;  
实际实现中通常不会逐像素全图搜索, 而是先估计这个椭圆会覆盖哪些 tile。

如果 tile 大小设为 $B x B$, 那么图像会被分成:

* $num_tiles_h = ceil(H / B)$
* $num_tiles_w = ceil(W / B)$

### 3. 分配到 tile

对每个高斯, 记录它会影响哪些 tile。  
这样后面每个 tile 只需要处理少量相关高斯, 渲染效率会高很多。

### 输出

* 可见高斯索引 `visible_ids`: $(N_vis,)$
* 每个 tile 对应的高斯列表 `tile_gaussians`

---

## Step 5: 深度排序

3DGS 的前向渲染通常要考虑前后遮挡关系;  
因此在一个 tile 或一个像素内, 通常会按照深度从前到后或从后到前进行累计。

### 输入

* `visible_ids`: $(N_vis,)$
* `depth`: $(N_vis,)$

### 核心思想

如果两个高斯都投影到了同一个像素附近:

* 更靠前的高斯应该先贡献颜色和透明度
* 更靠后的高斯会被前面的透射率衰减

所以一般会对相关高斯按深度排序。

### 输出

* 排序后的高斯索引 `sorted_ids`

---

## Step 6: 每个高斯在像素平面上的 splatting 权重

现在已经知道某个高斯会影响某些像素, 下一步就是计算它对这些像素到底贡献多少。

### 输入

* 某个高斯的 2D 中心 $mu_2d = (u_i,v_i)$
* 某个高斯的 2D 协方差 `Sigma_2d`: $(2,2)$
* 某个像素中心 $p = (u,v)$

### 1. 2D 高斯值

记像素相对高斯中心的偏移为:

$$
\Delta \mathbf{p}=
\begin{bmatrix}
u-u_i \\
v-v_i
\end{bmatrix}
$$

则这个像素处的 2D 高斯权重可以写成:

$$
g_i(u,v)=\exp\left(
-\frac{1}{2}
\Delta \mathbf{p}^T
(\Sigma_i^{2D})^{-1}
\Delta \mathbf{p}
\right)
$$

这个值越大, 说明像素越靠近该高斯在屏幕空间中的中心, 因而贡献越大。

### 2. 结合不透明度

像素处的有效 alpha 通常写成:

$$
\tilde\alpha_i(u,v)=\alpha_i \cdot g_i(u,v)
$$

其中:

* $alpha_i$ 控制这个高斯整体有多不透明
* $g_i(u,v)$ 控制这个像素是否真的落在这个高斯的主要影响范围内

### 输出

* 某个像素处每个高斯的局部 alpha: $alpha_tilde$

---

## Step 7: alpha blending 合成像素颜色

这一步和 NeRF 的体渲染思想很像;  
区别在于 NeRF 是沿射线对采样点积分, 3DGS 是对投影到屏幕空间的高斯做前到后的 alpha 合成。

### 输入

* 当前像素相关高斯的颜色 `c_i`: $(3,)$
* 当前像素相关高斯的有效 alpha `alpha_tilde_i`
* 深度排序结果

### 1. 透射率

设这些高斯已经按前到后排序, 那么第 $i$ 个高斯之前的透射率为:

$$
T_i=\prod_{j=1}^{i-1}(1-\tilde\alpha_j)
$$

### 2. 像素颜色

最终像素颜色为:

$$
\hat{\mathbf{C}}(u,v)=\sum_i T_i\tilde\alpha_i\mathbf{c}_i
$$

如果考虑背景色 $c_bg$, 则常写成:

$$
\hat{\mathbf{C}}(u,v)=
\sum_i T_i\tilde\alpha_i\mathbf{c}_i
+
T_{end}\mathbf{c}_{bg}
$$

其中：

$$
T_{end}=\prod_i(1-\tilde\alpha_i)
$$

### 输出

* 单个像素颜色 `rgb`: $(3,)$
* 整张图像 `image`: $(H,W,3)$

---

## Step 8: 颜色参数通常不是直接 RGB, 而是 SH

3DGS 常见实现里, 高斯颜色并不是简单存一个固定 RGB;  
而是存一组球谐函数系数, 这样颜色就可以随视角变化。

### 输入

* 高斯的 SH 系数 `sh_i`: $(C_sh,)$
* 观察方向 `d`: $(3,)$

### 公式

颜色可以写成:

$$
\mathbf{c}_i(\mathbf{d})=\sum_{m} a_{im} Y_m(\mathbf{d})
$$

其中：

* $a_im$ 是第 $i$ 个高斯的 SH 系数
* $Y_m(d)$ 是对应阶数的球谐基函数

如果不考虑视角相关颜色, 也可以简单理解成直接学习一个 RGB。

### 输出

* 当前视角下该高斯的颜色 `c_i(d)`: $(3,)$

---

## Step 9: 训练目标

训练时, 给定当前相机视角对应的 GT 图像 $I_gt$,  
渲染器输出预测图像 $I_pred$, 然后做监督。

### 输入

* `image_pred`: $(H,W,3)$
* `image_gt`: $(H,W,3)$

### 常见损失

最基本的是像素级重建损失:

$$
\mathcal{L}_{rgb}=\left\|\hat I-I\right\|_2^2
$$

实际 3DGS 常见还会加:

* $L1$ 损失
* $SSIM$ 损失
* 一些正则项, 比如对尺度或不透明度的约束

可以概括成:

$$
\mathcal{L}=
\lambda_1\mathcal{L}_{L1}
+
\lambda_2\mathcal{L}_{SSIM}
+
\lambda_3\mathcal{L}_{reg}
$$

### 输出

* 标量损失 `loss`: $(1,)$

---

## Step 10: densification 和 pruning

3DGS 训练里很关键的一点是:  
高斯的数量不是固定死的, 训练过程中通常会动态增删。

### densification

如果某些区域:

* 重建误差大
* 梯度大
* 当前高斯太稀疏

那就会在这些地方复制、分裂或新增高斯, 让表示更细。

### pruning

如果某些高斯:

* alpha 很低
* 贡献很小
* 屏幕空间几乎看不见

那就可以删掉, 减少冗余计算。

所以 3DGS 训练不是单纯"更新参数";  
而是同时在做:

* 高斯参数优化
* 高斯集合结构调整

---

## 从“单个高斯”到“整张图片”的函数链路

如果按推理时渲染整张图来理解, 整个维度流可以记成：

1. 输入场景高斯
   $mu: (N_g,3)$, $Sigma: (N_g,3,3)$, $alpha: (N_g,)$, $color/sh: (N_g,...)$
2. 变换到相机坐标系
   $mu_cam: (N_g,3)$, $Sigma_cam: (N_g,3,3)$
3. 投影到图像平面
   $mean_2d: (N_g,2)$, $cov_2d: (N_g,2,2)$, $depth: (N_g,)$
4. 可见性筛选和 tile 划分
   $visible_ids: (N_vis,)$
5. 深度排序
   `sorted_ids`
6. 计算每个高斯在像素处的 2D Gaussian 权重
   $g_i(u,v)$ 和 $alpha_tilde_i(u,v)$
7. alpha blending
   $image: (H,W,3)$

---

## 你可以这样理解 3DGS 的两个“函数层级”

### 层级 1: 整个系统

$$
(K,T_{c2w},\mathcal{G}) \rightarrow I
$$

这是“输入相机和场景表示, 输出图像”的外部视角。

### 层级 2: 真正学习的表示

$$
\mathcal{G}=\{(\mu_i,\Sigma_i,\alpha_i,c_i)\}_{i=1}^{N_g}
$$

这是“场景由很多个高斯组成”的内部视角。

两者之间的桥梁就是:

* 3D 高斯到 2D 椭圆高斯的投影
* 屏幕空间 splatting
* alpha blending 合成

---
