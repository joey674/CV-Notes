# 3DGS
<!-- ############################### -->
## referrence
https://www.bilibili.com/video/BV1zi421v7Dr/?spm_id_from=333.788.recommend_more_video.-1&trackid=web_related_0.router-related-2479604-6dnm7.1773423875441.190&vd_source=84ae2dc9d7d25fd8637002a2bb332c48

<!-- ############################### -->
## background

**主动/被动渲染**
NeRF 中的 ray-casting 更接近一种被动渲染: 已知相机位姿后, 从像素出发发射射线, 再沿着射线去查询场景在这些位置的颜色和密度;  
也就是说, 它的前向过程是:

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

---
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

---
**3D高斯椭球性质**

Gaussian 椭球有非常好的数学性质。

从概率论角度, 一个 $k$ 维高斯密度可以写成:

$$
\mathcal{N}(\mathbf{x};\mathbf{p},\Sigma)=
\frac{1}{\sqrt{(2\pi)^k|\Sigma|}}
\exp\left(
-\frac{1}{2}
(\mathbf{x}-\mathbf{p})^T\Sigma^{-1}(\mathbf{x}-\mathbf{p})
\right)
$$

在 3DGS 里, $k=3$, 它在空间中的是一个椭球;

其中:

* $\mathbf{p}$ 是均值, 表示高斯中心
* $\Sigma$ 是协方差矩阵, 决定高斯的尺度、拉伸和朝向
* $|\Sigma|$ 是协方差矩阵的行列式
* $\mathbf{x}$ 是某个点的3维位置; 如果代入这个点到高斯分布里, 就能得到这个这个高斯分布对这个点的影响力 $G(\mathbf{x})$
* $G(\mathbf{x})$ 在 $[0,1]$ 之间分布, 代表影响; 比如 $\mathbf{x}$ 离高斯椭球很近, 那 $G(\mathbf{x})$ 就会接近1



**仿射变换后仍然保持高斯形式**

这是最关键的性质之一。

如果一个随机变量满足:

$$
\mathbf{x}\sim\mathcal{N}(\mathbf{p},\Sigma)
$$

那么经过仿射变换

$$
\mathbf{y}=A\mathbf{x}+\mathbf{b}
$$

之后, 仍然有

$$
\mathbf{y}\sim\mathcal{N}(A\mathbf{p}+\mathbf{b},A\Sigma A^T)
$$

这意味着:

* 高斯中心会跟着线性变换和平移一起变化
* 协方差也会按照矩阵乘法规则同步变化

所以高斯非常适合做"从 3D 到 2D 的可微投影"。

***

---
**结论**

* **高斯可以很好地表示一个 3D 椭球**
* **任何高斯椭球都可以看作是标准高斯球经过仿射变换得到**

---
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

---
**结论**

* **高斯可以很好的表示一个3维椭球; 高斯均值表示椭球中心, 协方差矩阵显式表示旋转和缩放;** 
* **这样在任意位置, 任意形状的椭球, 用高斯就可以完全表示**


<!-- ############################### -->
## Overview
可以把 3DGS 分成两个层次来看:

1. 它是一种显式的 3D 场景表征: 用很多带位置、形状、透明度和颜色的 Gaussian 来表示场景
2. 它的渲染管线是可微的: 可以把渲染误差反向传播回这些 Gaussian 参数

如果把训练数据写成一个数据集:

$$
\mathcal{R}\big(\mathcal{G},K,T_{w2c},H,W\big)\longrightarrow \hat I\in\mathbb{R}^{H\times W\times 3}
$$

其中:

* $N_v$ 是视角数量
* 第 $n$ 张 RGB 图像记为 $I_n\in\mathbb{R}^{H_n\times W_n\times 3}$
* 相机内参记为 $K_n\in\mathbb{R}^{3\times 3}$
* 世界坐标到相机坐标的外参记为 $T_{w2c}^{(n)}\in\mathbb{R}^{4\times 4}$

如果所有图片分辨率一致, 整个图像张量也可以记成 $\mathrm{images}\in\mathbb{R}^{N_v\times H\times W\times 3}$。

3DGS 学到的场景表示可以写成:

$$
\mathcal{G}=\left\{(\mu_i,q_i,s_i,o_i,f_i)\right\}_{i=1}^{M}
$$

其中:

* $M$ 是当前场景中 Gaussian 的数量
* $\mu_i\in\mathbb{R}^3$ 是第 $i$ 个 Gaussian 的中心
* $q_i\in\mathbb{R}^4$ 是旋转四元数
* $s_i\in\mathbb{R}^3$ 是各轴尺度
* $o_i\in\mathbb{R}$ 是 opacity
* $f_i$ 是颜色特征; 如果采用 SH, 通常有 $f_i\in\mathbb{R}^{3(L+1)^2}$; 如果不用 SH, 最简单时也可以把它看成 $f_i\in\mathbb{R}^3$

所以渲染器本身可以写成:

$$
\mathrm{Render}_{\mathcal{G}}:(K,T_{w2c},H,W)\longrightarrow \hat I\in\mathbb{R}^{H\times W\times 3}
$$

**整体流程**

而一次训练前向过程则是:

$$
(K,T_{w2c},H,W,\mathcal{G})
\rightarrow
\{\mu_{c,i},\Sigma_{c,i}\}_{i=1}^{M}
\rightarrow
\{u_i,\Sigma_{2D,i}\}_{i=1}^{M}
\rightarrow
\hat I
$$

直观上就是:

1. 先用多视角图像、位姿和内参把场景初始化成 Gaussian 场景
2. 再把这些 Gaussian 从当前相机视角投影到像素平面, 得到渲染图像
3. 最后拿渲染图和 GT 做损失, 反向更新 Gaussian 参数; 训练好后再渲染新视角图像

---
## Step1 初始化成高斯场景

3DGS 一开始把场景写成一堆显式的 3D Gaussian。

**输入**

训练集通常包含:

* 图像张量可以记为 $\mathrm{images}\in\mathbb{R}^{N_v\times H\times W\times 3}$
* 位姿张量可以记为 $\mathrm{poses}\in\mathbb{R}^{N_v\times 4\times 4}$
* 内参张量可以记为 $\mathrm{intrinsics}\in\mathbb{R}^{N_v\times 3\times 3}$, 如果所有相机共享内参, 也可以直接记成 $K\in\mathbb{R}^{3\times 3}$

单个视角的数据分别记为 $I_n\in\mathbb{R}^{H\times W\times 3}$、$T_{w2c}^{(n)}\in\mathbb{R}^{4\times 4}$ 和 $K_n\in\mathbb{R}^{3\times 3}$。

原始 3DGS 实践里通常会先用 COLMAP / SfM 得到稀疏点云, 其中点坐标可以记为 $\mathrm{points\_xyz}\in\mathbb{R}^{M_0\times 3}$, 点颜色可以记为 $\mathrm{points\_rgb}\in\mathbb{R}^{M_0\times 3}$。


**从稀疏点到 Gaussian**

每个稀疏 3D 点都会初始化成一个 Gaussian。  
批量看时, 高斯中心可以记为 $\mu\in\mathbb{R}^{M_0\times 3}$, 旋转可以记为 $q\in\mathbb{R}^{M_0\times 4}$, 尺度可以记为 $s\in\mathbb{R}^{M_0\times 3}$, 透明度可以记为 $o\in\mathbb{R}^{M_0\times 1}$, 颜色特征可以记为 $f\in\mathbb{R}^{M_0\times C_f}$。  
如果采用 SH, 那么特征维度通常满足 $C_f=3(L+1)^2$。

其中中心直接来自点云位置:

$$
\mu_i\in\mathbb{R}^3
$$

协方差由旋转和缩放参数化:

$$
\Sigma_i=R(q_i)S(s_i)S(s_i)^TR(q_i)^T
$$

所以批量的协方差也可以记为 $\Sigma\in\mathbb{R}^{M_0\times 3\times 3}$。  
如果特征用 SH 表示颜色, 那么观察方向满足 $\mathbf{d}\in\mathbb{R}^3$, 并且颜色可以写成 $\mathbf{c}_i(\mathbf{d})\in\mathbb{R}^3$。


**输出**

这一步结束后, 场景就从“稀疏点云”变成了“高斯场景表示”:

$$
\mathcal{G}_0=\left\{(\mu_i,q_i,s_i,o_i,f_i)\right\}_{i=1}^{M_0}
$$

## Step2 从高斯到像素

观察变换 $\rightarrow$ 投影变换 $\rightarrow$ 视口变换 / splatting。
<figure><img src="/nvs/assets/transformation.png" alt=""><figcaption></figcaption></figure>


**输入**

训练时先取一个视角, 它对应的真值图像、内参和外参分别记为 $I_{gt}\in\mathbb{R}^{H\times W\times 3}$、$K\in\mathbb{R}^{3\times 3}$ 和 $T_{w2c}\in\mathbb{R}^{4\times 4}$。  
同时取当前高斯场景参数 $\mu\in\mathbb{R}^{M\times 3}$、$\Sigma\in\mathbb{R}^{M\times 3\times 3}$、$o\in\mathbb{R}^{M\times 1}$ 和 $f\in\mathbb{R}^{M\times C_f}$。


**观察变换: 世界坐标到相机坐标**

设当前外参满足 $T_{w2c}=[R|t]$, 其中 $R\in\mathbb{R}^{3\times 3}$, $t\in\mathbb{R}^{3}$。

对单个 Gaussian, 中心和协方差会变成:

$$
\mu_{c,i}=R\mu_i+t
\in\mathbb{R}^3
$$

$$
\Sigma_{c,i}=R\Sigma_iR^T
\in\mathbb{R}^{3\times 3}
$$

批量看时, 变换后的中心和协方差可以分别记为 $\mu_{\mathrm{cam}}\in\mathbb{R}^{M\times 3}$ 和 $\Sigma_{\mathrm{cam}}\in\mathbb{R}^{M\times 3\times 3}$。


**投影变换: 3D Gaussian 到 2D Gaussian**

记

$$
\mu_{c,i}=(x_i,y_i,z_i)^T
$$

那么它就是这个高斯中心在当前相机下的位置。

---

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
\left.\frac{\partial \pi}{\partial \mathbf{x}}\right|_{\mu_{c,i}}
\in\mathbb{R}^{2\times 3}
$$

$$
\Sigma_{2D,i}=J_i\Sigma_{c,i}J_i^T
\in\mathbb{R}^{2\times 2}
$$

批量看时, 投影后的屏幕中心可以记为 $\mathbf{u}\in\mathbb{R}^{M\times 2}$, 深度可以记为 $\mathbf{z}\in\mathbb{R}^{M\times 1}$, 对应的 2D 协方差可以记为 $\Sigma_{2D}\in\mathbb{R}^{M\times 2\times 2}$。

---
**Splatting 和 Alpha 合成**

记某个像素位置为:

$$
\mathbf{x}_{pix}=(u,v)^T\in\mathbb{R}^2
$$

那么单个 Gaussian 对这个像素的 2D 权重为:

$$
g_i(\mathbf{x}_{pix})=
\exp\left(
-\frac{1}{2}
(\mathbf{x}_{pix}-\mathbf{u}_i)^T
(\Sigma^{eff}_{2D,i})^{-1}
(\mathbf{x}_{pix}-\mathbf{u}_i)
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
\alpha_i(\mathbf{x}_{pix})=o_i\cdot g_i(\mathbf{x}_{pix})
$$

### 4. 按深度排序后做前向合成

对于同一个像素, 先把所有会影响它的 Gaussian 按深度从近到远排序。

定义在第 $i$ 个 Gaussian 之前, 光线剩余的透射率为:

$$
T_i(\mathbf{x}_{pix})=\prod_{j<i}\left(1-\alpha_j(\mathbf{x}_{pix})\right)
$$

那么该像素最终颜色为:

$$
\hat{\mathbf{C}}(\mathbf{x}_{pix})=
\sum_i
T_i(\mathbf{p})\alpha_i(\mathbf{p})\mathbf{c}_i(\mathbf{d})
\in\mathbb{R}^3
$$

把所有像素拼起来, 就得到渲染图像 $\hat I\in\mathbb{R}^{H\times W\times 3}$。

---
**输出**

这一步的最终输出就是渲染图像 $\hat I\in\mathbb{R}^{H\times W\times 3}$。

也就是说, 3DGS 的前向其实就是:  
一堆 3D Gaussian $\rightarrow$ 投影成 2D 椭圆 $\rightarrow$ 把颜色和透明度铺到像素上。

---
**Step3 训练与反向传播**

前两步只是在说“怎么从场景渲染出图”;  
这一步说的是“怎么把这些 Gaussian 学好”。

---
**输入**

渲染图像记为 $\hat I\in\mathbb{R}^{H\times W\times 3}$, 真值图像记为 $I_{gt}\in\mathbb{R}^{H\times W\times 3}$。

---
**图像损失**

原始 3DGS 常见地使用:

$$
\mathcal{L}=(1-\lambda)\mathcal{L}_{L1}+\lambda\mathcal{L}_{D-SSIM}
$$

其中:

* 像素级颜色差异
* 结构相似性

---
**反向传播更新参数**

梯度会回传到高斯参数 $\mu\in\mathbb{R}^{M\times 3}$、$q\in\mathbb{R}^{M\times 4}$、$s\in\mathbb{R}^{M\times 3}$、$o\in\mathbb{R}^{M\times 1}$ 和 $f\in\mathbb{R}^{M\times C_f}$。

所以训练本质上是在不断调整:

* Gaussian 放在哪里
* Gaussian 有多大、朝向如何
* Gaussian 有多透明
* 从不同方向看它应该是什么颜色

---
**Densify 和 Prune**

3DGS 的另一个关键点是 Gaussian 数量不是固定的。

### 1. densify 的依据是什么

实现里常见依据是:

* 某个 Gaussian 在 view-space 下的位置梯度持续较大

这通常意味着:

* **densify / split / clone**: 在误差大、细节不够的地方增加 Gaussian
* **prune**: 删除 opacity 太小、贡献太弱的 Gaussian

所以随着训练进行:

* 一开始是较粗的表示
* 后面会逐渐长出更多、更细的 Gaussian





