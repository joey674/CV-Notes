# 3DGS

## Reference

1. https://www.bilibili.com/video/BV1zi421v7Dr/?spm_id_from=333.788.recommend_more_video.-1&trackid=web_related_0.router-related-2479604-6dnm7.1773423875441.190&vd_source=84ae2dc9d7d25fd8637002a2bb332c48
2. https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
3. https://github.com/graphdeco-inria/gaussian-splatting

## background

**主动/被动渲染**

NeRF 中的 ray-casting 更接近一种被动渲染: 已知相机位姿后, 从像素出发发射射线, 再沿着射线去查询场景在这些位置的颜色和密度;
也就是说, 它的前向过程更像是:

$$
\text{pixel} \rightarrow \text{ray} \rightarrow \text{sample points} \rightarrow \text{color}
$$

这里所谓"被动", 不是说 NeRF 不学习场景, 而是说在渲染时, 每个像素都要等着射线去"问"场景: 这条路径上有什么。

3DGS 则更接近主动渲染: 场景先被表示成一堆显式的 3D Gaussian, 渲染时不是沿每条射线反复查询隐式函数, 而是直接计算这些高斯会如何投影到图像平面, 并主动把自己的影响"铺"到像素上:

$$
\text{Gaussians} \rightarrow \text{project to image} \rightarrow \text{splat} \rightarrow \text{image}
$$

所以可以直观理解成:

* NeRF: 从像素出发, 找哪些 3D 位置会影响这个像素
* 3DGS: 从场景中的每个高斯出发, 看它会影响哪些像素

这也是两者在计算形态上的最大区别:

* NeRF 的瓶颈主要在沿射线的密集采样和 MLP 查询
* 3DGS 的瓶颈主要在大量高斯的投影、排序和 alpha 合成

***

**泼溅**

这里的 splatting 可以先把它想成一种"往屏幕上盖印章"的过程。

如果场景里只有一个 3D 点, 投影到图像上往往只对应一个离散像素, 这会很稀疏, 也不稳定;
而 3DGS 不是把一个元素当作无限小的点, 而是把它当作一个有空间范围的 3D Gaussian。这样它投影到 2D 后, 就不是一个点, 而是一个 2D 椭圆形的影响区域。

于是渲染时做的事情就是:

1. 把 3D Gaussian 投影到当前图像平面
2. 得到它在 2D 上的中心和协方差
3. 对它覆盖到的像素, 按高斯权重分配颜色和透明度
4. 按深度顺序把多个高斯做 alpha blending

这就是 splatting 的核心:
不是"一个像素只对应一个点", 而是"一个高斯把自己的影响软性地分摊到周围一片像素上"。

这个表示有几个直接好处:

* 比点云 rasterization 更平滑, 不容易出现孔洞
* 比 NeRF 沿射线密集采样更高效
* 对反向传播友好, 因为投影、权重、合成过程都可以做成可微

所以 3DGS 常被理解成:
把 3D 场景表示成一组可以被直接 rasterize 的软粒子, 再用可微 splatting 来渲染。

***

**3D 高斯椭球性质**

Gaussian 椭球有非常好的数学性质。

**多维高斯的基本形式**

一个 $k$ 维高斯可以写成:

$$
G(\mathbf{x})=
\frac{1}{\sqrt{(2\pi)^k|\Sigma|}}
\exp\left(
-\frac{1}{2}
(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)
\right)
$$

在 3DGS 里, $k=3$, 它在空间中对应一个 3D 椭球。

其中:

* $\mu$ 是均值, 表示高斯中心
* $\Sigma$ 是协方差矩阵, 决定高斯的尺度、拉伸和朝向
* $|\Sigma|$ 是协方差矩阵的行列式
* $\mathbf{x}$ 是某个点的 3D 位置

这里有一个很容易混淆的点:

* 作为概率密度时, 上面的 $G(\mathbf{x})$ 不一定严格落在 $[0,1]$
* 但在 splatting 里, 更常用的是忽略归一化常数后的指数核

$$
\tilde G(\mathbf{x})=
\exp\left(
-\frac{1}{2}
(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)
\right)
$$

它的核心含义是:

* 离高斯中心越近, 权重越大
* 离得越远, 权重越快衰减

**仿射变换后仍然保持高斯形式**

这是最关键的性质之一。

如果一个随机变量满足:

$$
\mathbf{x}\sim\mathcal{N}(\mu,\Sigma)
$$

那么经过仿射变换

$$
\mathbf{y}=A\mathbf{x}+\mathbf{b}
$$

之后, 仍然有

$$
\mathbf{y}\sim\mathcal{N}(A\mu+\mathbf{b},A\Sigma A^T)
$$

这意味着:

* 高斯中心会跟着线性变换和平移一起变化
* 协方差也会按照矩阵乘法规则同步变化

所以高斯非常适合做"从 3D 到 2D 的可微投影"。

***

**结论**

* **高斯可以很好地表示一个 3D 椭球**
* **任何高斯椭球都可以看作是标准高斯球经过仿射变换得到**

***

**仿射变换与旋转/缩放矩阵的联系**

**从标准高斯出发**

先从最简单的标准高斯出发, 可以理解为一个处于原点位置、标准大小的圆球:

$$
\mathbf{x}\sim\mathcal{N}(\mathbf{0},I)
$$

这个高斯的协方差等于单位矩阵 $I$。如果再做一次仿射变换 $A\mathbf{x}+\mathbf{b}$, 那么新的协方差就是:

$$
\Sigma=AIA^T=AA^T
$$

**把仿射变换拆成旋转和缩放**

在 3DGS 里, 通常不会直接优化一个任意的 $A$, 而是把它拆成:

$$
A=RS
$$

其中:

* $R$ 是旋转矩阵, 负责方向
* $S$ 是缩放矩阵, 负责各轴尺度

于是:

$$
\Sigma=AA^T=(RS)(RS)^T
$$

再利用转置的乘法规则:

$$
(RS)^T=S^TR^T
$$

最终得到:

$$
\Sigma=RSS^TR^T
$$

这就是 3DGS 里常见的协方差参数化形式。

如果 $S$ 是对角矩阵, 它表示沿主轴方向的缩放;
再乘上 $R$ 之后, 就把这个轴对齐的椭球旋转到了任意方向。

**结论**

* $R$ 很直观地控制高斯椭球朝向
* $S$ 很直观地控制高斯椭球在各个方向上的大小
* $\Sigma=RSS^TR^T$ 天然是半正定的

所以 3DGS 不需要直接去学一个自由的 $(3,3)$ 协方差矩阵, 而是学:

* 一个旋转参数
* 一个缩放参数

然后再把它们组合成协方差矩阵。

在实现里通常还会进一步约束:

* 缩放参数先在 log-space 中优化, 再通过 $\exp(\cdot)$ 变成正数
* 旋转参数常用四元数 $q\in\mathbb{R}^4$, 再归一化成单位四元数

于是单个 Gaussian 更常见的参数写法是:

$$
G_i=(\mu_i,q_i,s_i,o_i,f_i)
$$

其中:

* $\mu_i\in\mathbb{R}^3$: 位置
* $q_i\in\mathbb{R}^4$: 旋转四元数
* $s_i\in\mathbb{R}^3$: 各轴缩放参数
* $o_i\in\mathbb{R}$: opacity 参数
* $f_i$: 颜色特征, 通常是 SH 系数

***

**球谐函数表达颜色**

<figure><img src="../.gitbook/assets/{B08F76CA-880D-403F-B12F-2687EF2699EA}.png" alt="球谐函数表达方向相关颜色"></figure>

在 3DGS 里, 每个 Gaussian 不只需要一个"固有颜色", 还需要一个**从不同方向看过去时的颜色变化规律**。

如果把观察方向记成单位向量 $\mathbf{d}$, 那么颜色可以写成:

$$
\mathbf{c}_i(\mathbf{d})=
\sum_{l=0}^{L}\sum_{m=-l}^{l}
\mathbf{f}_{i,lm}Y_l^m(\mathbf{d})
$$

其中:

* $Y_l^m(\mathbf{d})$ 是球谐基函数
* $\mathbf{f}_{i,lm}$ 是需要学习的系数
* $L$ 是球谐阶数

球谐函数在这里的直觉是:

* $l=0$ 阶只表示常数项, 相当于"这个点的基础颜色"
* 更高阶会逐渐引入方向变化, 例如高光、镜面趋势、视角相关颜色

如果最高阶是 $L$, 那么每个颜色通道一共需要:

$$
(L+1)^2
$$

个系数。

例如官方实现中常见最大阶数是 $L=3$, 那么每个通道有 $16$ 个系数, RGB 一共是 $48$ 个系数。

这也是图里真正想表达的关键:

* 输入是一个方向 $\mathbf{d}$
* 球面基函数对这个方向做响应
* 再线性组合成当前视角下的颜色

因此 3DGS 不是简单给每个 Gaussian 存一个 RGB, 而是存了一个**可视角变化的颜色函数**。



## Overview

可以把 3DGS 分成两个层次来看:

1. **训练阶段**: 用多视角图片和相机参数, 学到一组能够表示场景的 3D Gaussians
2. **渲染阶段**: 给定一个目标相机, 把这些 3D Gaussians 投影到 2D 图像平面, 再 splat 成最终图像

<figure><img src="../.gitbook/assets/{902ADBE7-C353-4D6C-90B2-5AB3840BA6C6}.png" alt="3DGS整体流程"></figure>

和"生成点云 + 渲染点云"有点像;
只是这里的点不是普通离散点, 而是带有尺度、朝向、透明度和方向相关颜色的高斯椭球。

从函数形式上, 可以把 3DGS 渲染器写成:

$$
\mathcal{R}\big(\mathcal{G},K,T_{w2c},H,W\big)\longrightarrow \hat I\in\mathbb{R}^{H\times W\times 3}
$$

其中:

* $\mathcal{G}=\{(\mu_i,q_i,s_i,o_i,f_i)\}_{i=1}^{M}$ 是当前场景中的 $M$ 个 Gaussian
* $K$ 是相机内参
* $T_{w2c}$ 是世界坐标到相机坐标的外参
* $(H,W)$ 是目标图像分辨率
* $\hat I$ 是渲染图像

整个流程可以分成:

1. 相机标定和 SfM 得到稀疏点云
2. 用稀疏点云初始化 Gaussian 集合
3. 把 3D Gaussian 投影到当前图像平面
4. 用 splatting 和 alpha blending 渲染图片
5. 与 GT 图像计算损失
6. 反向传播更新高斯参数
7. 训练中穿插 densify / split / clone / prune

**训练中的 forward = 可微分渲染**
**推理/展示时的 render = 只渲染, 不回传梯度**



## 前向可微渲染: Step1 初始化成高斯场景

这一步的目标是:
先把"一堆多视角图像"变成"一组可优化的 Gaussian 参数"。

### 输入

* 多视角图像 $\{I_n\}_{n=1}^{N}$
* 相机内参和外参 $\{K_n,T_{w2c,n}\}_{n=1}^{N}$
* 由 SfM / COLMAP 恢复出的稀疏点云 $\mathcal{P}=\{\mathbf{p}_i,\mathbf{rgb}_i\}_{i=1}^{M_0}$

### 1. 先用 SfM 获得一个粗几何

3DGS 并不是从完全空白开始拟合。

通常先通过 COLMAP 这类 SfM 方法得到:

* 每张图像的相机位姿
* 场景里的稀疏 3D 点

这一步给了 3DGS 一个非常重要的几何先验:

* 哪些位置大概率属于真实表面
* 相机已经在什么位置拍过这些点

因此 3DGS 一开始就不是盲猜整个 3D 空间, 而是从一个已经对齐好的稀疏场景出发。

### 2. 从稀疏点初始化 Gaussian

对于每个稀疏点 $\mathbf{p}_i$, 初始化一个 Gaussian:

$$
G_i=(\mu_i,q_i,s_i,o_i,f_i)
$$

常见初始化方式是:

* $\mu_i=\mathbf{p}_i$, 直接把点的位置当作高斯中心
* $q_i=(1,0,0,0)$, 初始旋转设成单位四元数
* $s_i$ 根据邻域点间距初始化, 让初始高斯大小与点云密度匹配
* $o_i$ 初始化为较小但非零的 opacity
* $f_i$ 由点颜色初始化, 先把 RGB 写到 SH 的 DC 项, 其余高阶项先设为 0

如果把所有 Gaussian 收集起来, 初始场景可以写成:

$$
\mathcal{G}_0=
\left\{
(\mu_i,q_i,s_i,o_i,f_i)
\right\}_{i=1}^{M_0}
$$

这里官方实现里有两个比较值得记住的细节:

* 初始缩放常由最近邻距离决定, 这样稀的地方高斯更大, 密的地方高斯更小
* 初始 opacity 常取一个较小值, 比如有效 opacity 约为 $0.1$

### 3. 为什么这一步很重要

如果没有这个初始化, 3DGS 需要同时解决:

* 场景几何在哪里
* 每个高斯该多大
* 每个高斯该是什么颜色

这会让优化极不稳定。

而有了 SfM 点云之后:

* 位置先验已经比较靠谱
* 优化可以更快收敛
* 后续 densify 也更容易只在需要细化的地方增长

### 输出

* 初始 Gaussian 集合

$$
\mathcal{G}_0=
\left\{
(\mu_i,q_i,s_i,o_i,f_i)
\right\}_{i=1}^{M_0}
$$

这就是后续可微渲染真正要优化的对象。

## 前向可微渲染: Step2 坐标变换

这一步描述了一个 3D Gaussian 椭球如何投影到图像上。

<figure><img src="../.gitbook/assets/{A569CC54-339B-467D-BD24-3301549DAD08}.png" alt="3D高斯到2D椭圆的投影"></figure>

对于高斯椭球, 真正需要被投影的不是单独一个点, 而是:

* 高斯中心 $\mu_i$
* 3D 协方差 $\Sigma_i$

### 输入

* 当前相机内参 $K$
* 当前相机外参 $T_{w2c}=[R_{w2c}\ |\ \mathbf{t}]$
* 第 $i$ 个 Gaussian 的参数 $(\mu_i,q_i,s_i)$

### 1. 先恢复 3D 协方差矩阵

由旋转和缩放参数恢复 3D 协方差:

$$
\Sigma_i=R(q_i)S(s_i)S(s_i)^TR(q_i)^T
$$

如果把缩放写成对角阵:

$$
S(s_i)=
\mathrm{diag}\big(\exp(s_{i,1}),\exp(s_{i,2}),\exp(s_{i,3})\big)
$$

那么就能保证尺度始终为正。

### 2. 世界坐标到相机坐标

先把高斯中心变到相机坐标系:

$$
\mu_i^{cam}=R_{w2c}\mu_i+\mathbf{t}
$$

如果把 $\mu_i^{cam}$ 写成:

$$
\mu_i^{cam}=(x_i,y_i,z_i)^T
$$

那么它就是这个高斯中心在当前相机下的位置。

### 3. 中心点的透视投影

高斯中心投影到像素平面的公式是:

$$
u_i=f_x\frac{x_i}{z_i}+c_x
$$

$$
v_i=f_y\frac{y_i}{z_i}+c_y
$$

所以 3D 中心最终在图像上的 2D 中心为:

$$
\mathbf{u}_i=(u_i,v_i)^T
$$

### 4. 为什么要引入 Jacobian

如果只是投影一个点, 上面两条公式就够了;
但这里要投影的是一个 3D 椭球, 也就是要知道:

* 这个高斯在屏幕上会拉成什么形状
* 会覆盖多大的区域

麻烦在于透视投影不是全局线性的。

所以 3DGS 的做法是:
在高斯中心附近, 用投影函数的一阶泰勒展开做局部线性近似。

这时候投影 Jacobian 可以写成:

$$
J_i=
\begin{bmatrix}
\dfrac{f_x}{z_i} & 0 & -\dfrac{f_xx_i}{z_i^2}\\
0 & \dfrac{f_y}{z_i} & -\dfrac{f_yy_i}{z_i^2}
\end{bmatrix}
\in\mathbb{R}^{2\times 3}
$$

它表达的是:

* 3D 位置在 $x,y,z$ 上的微小变化
* 会如何映射成 2D 屏幕位置上的变化

### 5. 3D 协方差投影到 2D 协方差

令 $W=R_{w2c}$ 表示世界到相机的线性部分, 那么高斯在图像平面上的 2D 协方差可以近似写成:

$$
\Sigma_{2D,i}=J_iW\Sigma_iW^TJ_i^T
$$

这就是 Step2 最核心的式子。

它的含义是:

* 先把 3D 椭球变到当前相机坐标系
* 再用 Jacobian 近似透视投影
* 最终得到一个 2D 椭圆

所以一个 3D Gaussian 在屏幕上就不再是一个点, 而是:

$$
(\mathbf{u}_i,\Sigma_{2D,i})
$$

对应的一块 2D 椭圆 footprint。

实际实现里通常还会再做一个很小的低通补偿, 让每个高斯至少覆盖接近一个像素的范围, 避免 footprint 过小带来的 aliasing。

### 6. 从 2D 协方差得到覆盖区域

有了 $\Sigma_{2D,i}$ 之后, 就能求出椭圆主轴方向和大小。

实现里常见做法是:

* 先对 $\Sigma_{2D,i}$ 求特征值
* 再取一个大约 $3\sigma$ 的包围范围
* 用这个范围去算它会覆盖到哪些像素块或 tile

因此 Step2 的结果不仅仅是一个 2D 中心点, 而是:

* 屏幕中心位置
* 2D 椭圆形状
* 深度顺序
* 覆盖到哪些 tile

### 输出

对第 $i$ 个 Gaussian, Step2 之后可以得到:

* 2D 中心 $\mathbf{u}_i\in\mathbb{R}^2$
* 2D 协方差 $\Sigma_{2D,i}\in\mathbb{R}^{2\times 2}$
* 深度 $z_i$
* 屏幕 footprint 和对应包围盒

***

## 前向可微渲染: Step3 Splatting 和 Alpha 合成

这一步描述了这些高斯椭球最终是如何叠加渲染成一张图片的。

<figure><img src="../.gitbook/assets/{810DF6DC-DAA4-4A46-8F4D-B8A6BD52283C}.png" alt="Splatting和Alpha合成"></figure>

### 输入

* 第 $i$ 个 Gaussian 的 2D 中心 $\mathbf{u}_i$
* 第 $i$ 个 Gaussian 的 2D 协方差 $\Sigma_{2D,i}$
* 第 $i$ 个 Gaussian 的 opacity 参数 $o_i$
* 第 $i$ 个 Gaussian 的 SH 颜色系数 $f_i$
* 当前观察方向 $\mathbf{d}_i$

### 1. 先根据观察方向求颜色

3DGS 中一个高斯的颜色不是固定常数, 而是视角相关的:

$$
\mathbf{c}_i(\mathbf{d}_i)=
\sum_{l=0}^{L}\sum_{m=-l}^{l}\mathbf{f}_{i,lm}Y_l^m(\mathbf{d}_i)
$$

这里的 $\mathbf{d}_i$ 一般可以理解成:

* 从高斯中心指向当前相机中心的方向

也就是说, 同一个 Gaussian 在不同视角下, 可能会有不同的颜色输出。

### 2. 计算像素处的二维高斯权重

记某个像素位置为:

$$
\mathbf{p}=(u,v)^T\in\mathbb{R}^2
$$

那么单个 Gaussian 对这个像素的 2D 权重为:

$$
g_i(\mathbf{p})=
\exp\left(
-\frac{1}{2}
(\mathbf{p}-\mathbf{u}_i)^T
\Sigma_{2D,i}^{-1}
(\mathbf{p}-\mathbf{u}_i)
\right)
$$

它反映的是:

* 像素越接近高斯中心, 权重越大
* 偏离主轴越远, 权重越小

### 3. 乘上 opacity 得到 alpha

如果把有效 opacity 记成 $\sigma(o_i)$, 那么像素处的 alpha 常可以写成:

$$
\alpha_i(\mathbf{p})=\sigma(o_i)\cdot g_i(\mathbf{p})
$$

为了书写简洁, 很多笔记里也会直接把有效 opacity 记成 $o_i$, 写成:

$$
\alpha_i(\mathbf{p})=o_i\cdot g_i(\mathbf{p})
$$

### 4. 按深度排序后做前向合成

对于同一个像素, 先把所有会影响它的 Gaussian 按深度从近到远排序。

定义在第 $i$ 个 Gaussian 之前, 光线剩余的透射率为:

$$
T_i(\mathbf{p})=\prod_{j<i}\left(1-\alpha_j(\mathbf{p})\right)
$$

那么该像素最终颜色为:

$$
\hat{\mathbf{C}}(\mathbf{p})=
\sum_i
T_i(\mathbf{p})\alpha_i(\mathbf{p})\mathbf{c}_i(\mathbf{d}_i)
\in\mathbb{R}^3
$$

这个式子其实就是体渲染里 front-to-back compositing 的显式版本。

它表达了两个很重要的物理含义:

* 前面的高斯会先占据一部分透明度
* 后面的高斯只能在剩余透射率里继续贡献颜色

所以 3DGS 虽然不是 NeRF 那种沿射线积分, 但仍然保留了"遮挡"和"可见性"的建模。

### 5. 从单个像素扩展到整张图

对图像中每个像素都做上面的累加:

$$
\hat I=
\left\{
\hat{\mathbf{C}}(\mathbf{p})
\right\}_{\mathbf{p}\in\Omega}
\in\mathbb{R}^{H\times W\times 3}
$$

其中 $\Omega$ 是整个图像像素网格。

### 输出

* 渲染图像 $\hat I\in\mathbb{R}^{H\times W\times 3}$
* 训练时通常还会同时保留中间量:
    * 哪些 Gaussian 对哪些像素可见
    * 每个像素的透射率
    * 反向传播所需的中间缓存

## 前向可微渲染: Step4 并行化加速

3DGS 能做到实时渲染, 很大程度上靠的是它的 tile-based 并行 rasterization。

<figure><img src="../.gitbook/assets/{10A6FB2D-3A08-4562-AC4B-1C482A0CB555}.png" alt="3DGS并行化渲染流程"></figure>

### 为什么要并行化

如果暴力做 Step3, 对每个像素都遍历所有 Gaussian, 代价会非常大。

所以实现里会把图像分成很多 tile, 然后让:

* 每个 Gaussian 只去影响自己真正覆盖到的 tile
* 每个 tile 独立并行地完成内部渲染

官方 rasterizer 常见的 tile 大小是:

$$
16\times 16
$$

### 1. 预处理每个 Gaussian

在 Step2 之后, 每个 Gaussian 已经知道:

* 自己在屏幕上的 2D 中心
* 2D 协方差或 conic 参数
* 自己的深度
* 自己大致覆盖的屏幕包围盒

于是可以先算出:

* 它会与哪些 tile 重叠
* 每个 tile 里是否需要处理它

### 2. 复制到覆盖到的 tile

如果某个 Gaussian 覆盖了多个 tile, 那么实现里会为每个重叠 tile 生成一条记录。

也就是说, 后续处理的不是单纯的 Gaussian 列表, 而是:

* `(tile_id, depth, gaussian_id)` 这类键值对

这样做的好处是:

* 可以先按 tile 分组
* 再在每个 tile 内按深度排序

### 3. 按 tile 和 depth 排序

排序后, 列表结构就会自然变成:

* 先是同一个 tile 的所有 Gaussian
* 这些 Gaussian 在 tile 内又按深度从近到远排列

这样每个 tile 渲染时就不需要再做昂贵的全局检索。

### 4. 每个 tile 独立做 blend

随后 GPU 上每个 tile 都可以独立并行:

1. 读取自己负责的 Gaussian 范围
2. 对 tile 内像素计算高斯权重
3. 做前向 alpha blending
4. 输出本 tile 的像素结果

因为 tile 之间基本互不依赖, 这非常适合 GPU 并行执行。

### 5. 为什么比 NeRF 快很多

NeRF 常见计算模式是:

* 每条射线采样很多点
* 每个采样点都要查询一次 MLP
* 然后再做体渲染积分

而 3DGS 的计算模式是:

* 先把显式高斯投影到图像平面
* 只在真正受影响的局部区域做 rasterization
* 整个流程主要是矩阵运算、排序和并行 blend

这让它在渲染时非常接近传统图形学管线, 所以能做到实时。

## 反向梯度传播训练

前面的 Step1-Step4 是 forward。
训练时, 渲染结果还要和真实图像比较, 再把误差回传到每个 Gaussian 参数上。

<figure><img src="../.gitbook/assets/{754E8C22-2258-4192-9897-E642F9029E65}.png" alt="3DGS反向传播"><figcaption>这张图强调的是 3DGS 的可微性: 图像误差会沿着 alpha 合成、2D footprint、协方差和 SH 颜色一路回传, 最终更新高斯的位置、大小、旋转、透明度和颜色系数。</figcaption></figure>

### 1. 重建损失

常见的图像损失写成:

$$
\mathcal{L}=(1-\lambda)\mathcal{L}_{L1}+\lambda\mathcal{L}_{D\text{-}SSIM}
$$

其中:

* $\mathcal{L}_{L1}$ 约束逐像素颜色误差
* $\mathcal{L}_{D\text{-}SSIM}$ 更关注结构相似性

在官方实现里, 也常写成:

$$
\mathcal{L}=(1-\lambda)\| \hat I-I \|_1+\lambda(1-\mathrm{SSIM}(\hat I,I))
$$

一个常见默认值是:

$$
\lambda=0.2
$$

### 2. 梯度会回传到哪些参数

梯度会回传到高斯参数:

* $\mu\in\mathbb{R}^{M\times 3}$: 位置
* $q\in\mathbb{R}^{M\times 4}$: 旋转
* $s\in\mathbb{R}^{M\times 3}$: 缩放
* $o\in\mathbb{R}^{M\times 1}$: opacity
* $f$: SH 颜色系数

所以训练本质上是在不断调整:

* Gaussian 放在哪里
* Gaussian 有多大、朝向如何
* Gaussian 有多透明
* 从不同方向看它应该是什么颜色

### 3. 梯度为什么是有意义的

因为整个链条都是可微的:

* 3D 参数到 2D footprint 的投影近似可微
* 像素处的高斯权重可微
* alpha compositing 可微
* SH 颜色关于系数也是线性的, 很容易反传

所以图像误差可以直接指导:

* 高斯中心往哪里挪
* footprint 应该变大还是变小
* opacity 应该增强还是减弱
* 哪些方向上的颜色该怎么改

### 4. SH 阶数不是一开始就全开

3DGS 一个比较实用的训练技巧是:

* 先只用低阶 SH
* 随着训练稳定, 再逐步提升阶数

官方实现里常见做法是:

* 每 1000 次迭代把 active SH degree 加 1
* 最高到 degree 3

这样做的直觉是:

* 训练前期先把几何和粗颜色学稳
* 后期再逐渐加入更复杂的视角相关颜色

因此你原笔记里那句"前期更侧重点的位置信息, 后期再更准确显示颜色"是对的, 只是更准确地说:

* 前期限制了颜色函数的自由度
* 后期再慢慢放开方向相关颜色的表达能力

### 5. 训练并不是只优化, 还会改模型容量

3DGS 和普通固定参数量网络不一样。

训练过程中不仅参数在变, 模型里 Gaussian 的数量也在变:

* 表示不够时就 densify
* 贡献太弱时就 prune

这也是 3DGS 能同时兼顾粗结构和细节的重要原因。

***

**Densify 和 Prune**

<figure><img src="../.gitbook/assets/{8BC0F003-E4DC-44B5-AFBF-3C6AC958AD91}.png" alt="3DGS densify clone或split示意1"></figure>

<figure><img src="../.gitbook/assets/{C89EF1C5-D853-463C-B4C3-9FBF3F2E0CF8}.png" alt="3DGS densify clone或split示意2"></figure>

3DGS 的另一个关键点是 Gaussian 数量不是固定的。

### 1. densify 的依据是什么

实现里常见依据是:

* 某个 Gaussian 在 view-space 下的位置梯度持续较大

这通常意味着:

* 当前 Gaussian 放得不够准
* 或者它一个人无法表达局部细节

官方实现中, densify 常从第 500 次迭代后开始, 到第 15000 次迭代左右停止, 并且每隔 100 次迭代检查一次。

### 2. clone: 小高斯直接复制

如果一个 Gaussian:

* 梯度大
* 但自己本身尺度不大

那么更合理的做法往往是 clone:

* 直接复制一个新的 Gaussian
* 两者再在后续训练中各自微调

它更像是:

* "这个位置需要更高采样密度"

### 3. split: 大高斯拆成多个小高斯

如果一个 Gaussian:

* 梯度大
* 而且尺寸本身偏大

那往往说明:

* 它覆盖的区域太广了
* 一个大椭球没法同时描述里面的多个细节

这时更适合 split:

* 在原 Gaussian 附近采样出多个子 Gaussian
* 每个子 Gaussian 位置稍微偏移
* 缩小子 Gaussian 的尺度
* 然后把原来那个过大的 Gaussian 删除

所以 split 更像是:

* "把一个粗糙的大 blob 拆成几个细颗粒"

### 4. prune: 删掉没贡献或异常大的 Gaussian

训练过程中也会不断 prune。

常见删除条件包括:

* opacity 太小, 几乎不再贡献颜色
* 屏幕半径过大, 影响范围异常
* 世界尺度过大, 已经偏离合理场景结构

例如官方实现里, 一个典型的 opacity 阈值是:

$$
0.005
$$

### 5. opacity reset 的作用

3DGS 里还有一个常见技巧:
周期性把 opacity 压回比较小的值。

目的不是破坏训练, 而是:

* 防止某些 Gaussian 太早"锁死"成高 opacity
* 给后续 densify 和重新分配可见性留出空间

官方实现中, 常见 reset 周期是每 3000 次迭代一次。

### 6. 训练过程中场景是如何演化的

所以随着训练进行, 场景通常会经历这样的变化:

1. 一开始只有稀疏、比较粗的 Gaussian
2. 先拟合出大体几何和粗颜色
3. 在高误差区域 clone / split, 容量逐渐增加
4. 无用高斯被 prune 掉
5. 最后留下数量更合适、位置更准确、形状更细致的一组 Gaussian

这也是 3DGS 和固定大小参数模型一个很不一样的地方:

* 它不是只优化参数值
* 还在动态优化表示的结构和容量

***

**输出**

最终得到训练好的场景表示:

$$
\mathcal{G}^*=
\left\{
(\mu_i,q_i,s_i,o_i,f_i)
\right\}_{i=1}^{M^*}
$$

其中:

* $M^*$ 是训练结束后最终保留下来的 Gaussian 数量
* 它一般与初始稀疏点数 $M_0$ 不同, 因为中间做过 densify 和 prune

训练结束后, 如果给一个新的目标相机:

* $K_{new}\in\mathbb{R}^{3\times 3}$
* $T_{w2c,new}\in\mathbb{R}^{4\times 4}$
* 目标分辨率 $(H_{new},W_{new})$

那么重复 Step2-Step4 就可以得到:

$$
I_{new}\in\mathbb{R}^{H_{new}\times W_{new}\times 3}
$$

一句话总结 3DGS:

* **先把场景显式表示成一组可学习的 3D Gaussian**
* **再把这些 Gaussian 直接投影成 2D 椭圆并做可微 splatting**
* **训练中一边优化位置/形状/颜色, 一边动态增删 Gaussian**

所以它同时兼顾了:

* NeRF 那种接近连续场表示的质量
* 以及更接近传统图形学 rasterization 的实时渲染效率
