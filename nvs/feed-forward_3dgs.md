# Feed-Forward 3DGS
ref:
1. https://arxiv.org/abs/2312.12337
2. https://arxiv.org/abs/2403.14627
3. https://arxiv.org/abs/2404.19702
4. https://arxiv.org/abs/2402.05054
5. https://arxiv.org/abs/2408.13912
6. https://arxiv.org/abs/2410.24207

## Overview

原始 3DGS 的核心是:  
针对**单个场景**维护一组 Gaussian 参数, 然后用多视角图像反向传播去优化这些参数。

Feed-forward 3DGS 的核心则是:  
先在**大量场景**上训练一个网络, 让它学会从输入图像直接预测 Gaussian 场景表示;  
测试时只需要一次前向传播, 就可以得到该场景的 3DGS 表达。

如果输入视角已经带有相机参数, 那么前馈式 3DGS 可以抽象成:

$$
G_\phi:
\left\{
(I_n,K_n,T_{w2c}^{(n)})
\right\}_{n=1}^{N_s}
\longrightarrow
\mathcal{G}=
\left\{
(p_i,q_i,s_i,o_i,f_i)
\right\}_{i=1}^{M}
$$

然后再通过 3DGS 渲染器得到目标视角图像:

$$
\hat I_t=
\mathrm{Render}
(\mathcal{G},K_t,T_{w2c}^{(t)})
$$

其中:

* $N_s$ 是输入 source views 的数量
* $\phi$ 是前馈网络参数
* $\mathcal{G}$ 是网络直接预测出的 Gaussian 场景表示

如果把训练写成跨场景学习, 那么优化目标可以写成:

$$
\min_\phi
\sum_{\text{scene}}
\sum_{t}
\mathcal{L}
\left(
\mathrm{Render}(G_\phi(\mathcal{D}_{src}),K_t,T_{w2c}^{(t)}),
I_t
\right)
$$

也就是说:

* 原始 3DGS: 学的是某一个场景的 Gaussian 参数
* Feed-forward 3DGS: 学的是一个从图像到 Gaussian 的映射函数


### 整体流程

前馈式 3DGS 的典型流程可以概括成:

1. 输入少量 source views
2. 用图像编码器提取每个视角的 2D 特征
3. 融合多视角几何信息, 估计深度 / 对应关系 / 3D 结构
4. 直接预测 Gaussian 的位置、尺度、旋转、opacity 和颜色特征
5. 把预测出的 Gaussian 渲染到 target view
6. 用 target view 的监督信号训练整个网络

因此它保留了 3DGS 的**显式高斯表示**和**可微 splatting 渲染器**,  
但把原来针对单场景的 iterative optimization, 替换成了跨场景的 amortized inference。

---

## Step 1: 输入形式

### 1. Posed sparse-view 输入

最常见的设置是输入少量已经标定好的视角:

$$
\mathcal{D}_{src}=
\left\{
(I_n,K_n,T_{w2c}^{(n)})
\right\}_{n=1}^{N_s}
$$

其中:

* $I_n\in\mathbb{R}^{H\times W\times 3}$ 是第 $n$ 个输入图像
* $K_n\in\mathbb{R}^{3\times 3}$ 是内参
* $T_{w2c}^{(n)}\in\mathbb{R}^{4\times 4}$ 是外参

这一类方法通常直接利用相机几何来做 back-projection 和 multi-view fusion。

### 2. Unposed sparse-view 输入

另一类设置是不提供相机位姿, 也就是只输入图像:

$$
\mathcal{D}_{src}=
\left\{
I_n
\right\}_{n=1}^{N_s}
$$

这时网络除了预测 Gaussian, 还要处理:

* 相对位姿估计
* 尺度歧义
* 场景坐标系 / canonical frame 的建立

因此 pose-free feed-forward 3DGS 一般比 posed 设置更难。

### 3. Object-level 输入

在 object generation 或 single-image-to-3D 的场景中, 输入也常常不是原始多视角照片, 而是:

* 单张图像
* 文本生成的多视角图像
* 由大模型先合成出的多视图结果

然后再由 Gaussian 网络把这些图像转成 3DGS 表达。

---

## Step 2: 图像编码与几何特征

前馈式 3DGS 的第一步通常都是先把输入图像编码成 2D 特征:

$$
F_n=E_\phi(I_n)\in\mathbb{R}^{H'\times W'\times C}
$$

其中:

* $E_\phi$ 是图像编码器
* $H',W'$ 是降采样后的特征分辨率
* $C$ 是通道维度

仅有单视角特征还不够, 因为 Gaussian 的中心位置本质上是 3D 量。  
因此下一步通常要补几何信息, 常见方式有两类:

### 1. 显式深度 / 代价体

先构建每个像素在不同深度假设下的匹配代价, 再得到深度分布:

$$
P(d\mid u,v)
$$

然后求期望深度:

$$
\hat z(u,v)=\sum_d d\cdot P(d\mid u,v)
$$

或者直接取最大概率深度:

$$
\hat z(u,v)=\arg\max_d P(d\mid u,v)
$$

### 2. 隐式几何聚合

不显式构建深度体, 而是让网络通过:

* cross-view attention
* epipolar attention
* transformer token mixing
* recurrent fusion

直接在多视角特征之间建立 3D 对应关系。

无论采用哪一类, 目的都相同:  
把 2D 像素特征和 3D 结构联系起来, 为后面的 Gaussian 参数预测提供几何约束。

---

## Step 3: 从像素 / token 到 Gaussian 参数

前馈网络最终需要预测一组 Gaussian:

$$
\mathcal{G}=
\left\{
(p_i,q_i,s_i,o_i,f_i)
\right\}_{i=1}^{M}
$$

其中:

* $p_i\in\mathbb{R}^3$: Gaussian 中心
* $q_i\in\mathbb{R}^4$: 旋转四元数
* $s_i\in\mathbb{R}^3$: 各轴尺度
* $o_i\in\mathbb{R}$: opacity
* $f_i$: 颜色或 SH 特征

常见做法是:

### 1. 每个像素预测一个或多个 Gaussian

如果在特征图上每个位置 $(u,v)$ 预测 $K_g$ 个 Gaussian, 那么总数可以写成:

$$
M=N_s\cdot H'\cdot W'\cdot K_g
$$

也就是说, 每个 source view 的每个像素 / patch 都对应若干个 3D Gaussian。

### 2. 先预测深度, 再回投中心

如果中心是由深度回投得到, 那么相机坐标系下的 3D 点可写成:

$$
p_c=
\hat z
K^{-1}
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
$$

再通过位姿变换到世界坐标:

$$
p_w=T_{c2w}p_c,\qquad T_{c2w}=T_{w2c}^{-1}
$$

这种方式的优点是:

* 中心位置有明确几何意义
* 更容易利用多视角一致性
* 训练初期通常更稳定

### 3. 直接回归全部参数

也有方法直接让网络从多视角特征输出:

$$
(p_i,q_i,s_i,o_i,f_i)=D_\phi(\text{multi-view features})
$$

这种方式更灵活, 但对网络容量和数据分布更敏感。

### 4. 参数约束

和原始 3DGS 一样, 前馈模型也常常不会直接输出“已经合法”的参数, 而是输出无约束变量再做映射:

$$
S_i=\mathrm{diag}(\exp(\tilde s_i))
$$

$$
o_i=\sigma(\tilde o_i)
$$

$$
q_i=\frac{\tilde q_i}{\|\tilde q_i\|_2}
$$

于是协方差仍然写成:

$$
\Sigma_i=R(q_i)S_iS_i^TR(q_i)^T
$$

因此从表示形式上说, feed-forward 3DGS 和原始 3DGS 并没有本质区别;  
主要区别在于这些参数是**怎么得到的**。

---

## Step 4: 多视角融合

如果多个输入视角都会预测 Gaussian, 那么还需要把这些结果融合成同一个场景表示。

最简单的记法可以写成:

$$
\mathcal{G}=
\bigcup_{n=1}^{N_s}\mathcal{G}^{(n)}
$$

但实际实现通常不会只是直接并集, 还会考虑:

* 多视角重复预测的冗余
* 遮挡与可见性
* 不同视角的尺度偏差
* 相邻高斯的合并或筛除

在 posed 设置下, 融合通常是在统一世界坐标系中进行。  
在 unposed 设置下, 则常常要先建立一个 canonical frame, 再把所有 Gaussian 映射到同一坐标系。

---

## Step 5: 渲染与训练

得到 Gaussian 集合之后, 后续就和普通 3DGS 很接近了:

$$
\hat I_t=
\mathrm{Render}
(\mathcal{G},K_t,T_{w2c}^{(t)})
$$

训练时常见的损失可以写成:

$$
\mathcal{L} =
\mathcal{L}_{rgb}
+\lambda_{ssim}\mathcal{L}_{D-SSIM}
+\lambda_{geo}\mathcal{L}_{geo}
+\lambda_{pose}\mathcal{L}_{pose}
+\lambda_{reg}\mathcal{L}_{reg}
$$

其中不同方法使用的监督项并不完全相同, 常见的包括:

* RGB 重建损失
* SSIM / D-SSIM
* 深度监督
* 跨视角几何一致性
* 相机位姿监督或相对位姿监督
* Gaussian 数量、尺度或 opacity 的正则项

因此前馈式 3DGS 的训练方式是:

1. 输入 source views
2. 预测 Gaussian 场景表示
3. 在 target views 上渲染
4. 用 target views 的误差监督网络参数 $\phi$

注意这里优化的是**网络参数 $\phi$**, 而不是每个测试场景自己的 Gaussian 参数。

---

## Step 6: 为什么它比原始 3DGS 快

原始 3DGS 的耗时主要来自:

* 每个新场景都要从头优化
* 要进行上万次 iteration
* densify / prune 是在线进行的

Feed-forward 3DGS 把这部分成本转移到了离线训练阶段:

* 训练阶段: 在大量场景上学习通用映射
* 测试阶段: 一次前向直接输出 Gaussian

因此它的核心收益是:

* **测试时速度快**
* **不需要 per-scene optimization**
* **更适合交互式或在线场景重建**

但代价也很明显:

* 对训练数据分布依赖更强
* 泛化到新场景时可能不如优化式方法稳定
* 在困难材质、复杂遮挡、稀疏视角下, 质量常常仍落后于单场景精优化

---

## 常见技术路线

### 1. Posed sparse-view reconstruction

这一类方法输入的是少量已知位姿的图像, 目标是直接做 generalizable sparse-view reconstruction。

代表方法:

* **pixelSplat**: 从 image pairs 预测 3D Gaussian, 核心思想是 probabilistic per-pixel depth + multi-Gaussian prediction
* **MVSplat**: 更强调高效的 multi-view stereo 路线, 先构建几何特征再预测 Gaussian
* **GS-LRM**: 采用更大规模的 reconstruction model, 从 2-4 个 posed views 直接输出高质量 Gaussian

### 2. Pose-free feed-forward reconstruction

这一类方法不要求输入相机位姿, 网络需要同时解决几何和姿态问题。

代表方法:

* **Splatt3R**: 从 uncalibrated image pairs 零样本预测 Gaussian
* **NoPoSplat**: 从 sparse unposed images 直接恢复 Gaussian 场景表达

这一方向更贴近真实应用, 因为实际图像常常并没有现成的标定位姿。

### 3. Image-to-3D / text-to-3D

这一类方法常见于对象级生成任务:

* 先通过扩散模型或多视图生成模型得到一组一致的视角图像
* 再用 feed-forward Gaussian 网络直接生成 3DGS

代表方法:

* **LGM**: 大多视图 Gaussian 模型, 既可以做 image-to-3D, 也可以做 text-to-3D

---

## 与原始 3DGS 的差异

### 1. 学习对象不同

原始 3DGS 学的是:

$$
\mathcal{G}_{scene}
$$

也就是某个场景自己的 Gaussian 参数。

Feed-forward 3DGS 学的是:

$$
G_\phi
$$

也就是一个从图像到 Gaussian 的通用映射。

### 2. 训练单位不同

* 原始 3DGS: 单场景训练
* Feed-forward 3DGS: 跨场景训练

### 3. 推理方式不同

* 原始 3DGS: 先优化, 后渲染
* Feed-forward 3DGS: 先前向预测 Gaussian, 再直接渲染

### 4. 适用场景不同

* 原始 3DGS: 更适合高质量单场景重建
* Feed-forward 3DGS: 更适合实时、泛化式、批量式的 3D 重建

---

## 局限

Feed-forward 3DGS 虽然速度快, 但目前通常还会遇到下面这些问题:

1. **泛化瓶颈**  
训练集和测试集分布差距较大时, 质量下降明显。

2. **高斯数量难以自适应**  
很多方法会固定每像素输出 Gaussian 数量, 不像原始 3DGS 那样可以在训练中动态 densify。

3. **几何精度依赖多视角质量**  
输入视角太少、重叠太弱或位姿不准时, 很容易出现漂浮点、错误深度和结构粘连。

4. **局部细节不如优化式 3DGS**  
对于高频纹理、反射、透明体和复杂遮挡, 直接前向预测的表示往往还不够精细。


## 总结

Feed-forward 3DGS 可以看成是把原始 3DGS 的“单场景优化问题”改写成了“跨场景学习问题”。

核心变化不在于 Gaussian 表示本身, 而在于:

* 原始 3DGS 通过优化得到 Gaussian
* Feed-forward 3DGS 通过网络前向预测 Gaussian

因此它的主要价值是**速度和泛化**,  
而当前的主要挑战则是**精度、自适应能力和真实场景鲁棒性**。
