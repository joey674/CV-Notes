# NeRF
ref: 
1. https://www.bilibili.com/video/BV1dpLWzUE7K/?spm_id_from=333.1387.homepage.video_card.click&vd_source=84ae2dc9d7d25fd8637002a2bb332c48
2. https://www.bilibili.com/video/BV1CC411V7oq/?spm_id_from=333.1387.upload.video_card.click&vd_source=84ae2dc9d7d25fd8637002a2bb332c48
## Overview

**整个 NeRF 渲染系统**可以看成一个从相机参数到图像的函数;
**可学习的部分**而是中间那个 **3D 辐射场函数**;
然后通过输入多视角图片,可以认为是输入以及对应输出去训练这个神经网络函数拟合3D场景; 之后把新的想要的观测视角输入这个神经网络函数,就可以获取到对应的图片;

可以写成：

$$
\mathrm{NeRF}_\theta:(K,\ T_{c2w},\ H,\ W)\longrightarrow I\in\mathbb{R}^{H\times W\times 3}
$$

其中：

* $K$: $(3,3)$，相机内参
* `T_c2w`: $(4,4)$，相机外参，表示 camera-to-world 变换
* $H, W$: 图像高宽
* $I$: $(H,W,3)$，输出图像，每个像素是 RGB 向量

而可学习的 3D 辐射场函数是：

$$
F_\theta:(\mathbf{x},\mathbf{d})\longrightarrow (\mathbf{c},\sigma)
$$

其中：

* `x`: $(3,)$，场景中某个采样点的 3D 位置
* `d`: $(3,)$，这个采样点的被观察方向
* `c`: $(3,)$，这个采样点的颜色
* `sigma`: 标量，这个采样点的体密度

在辐射场的前后也是有对应的前处理和后处理来将系统输入输出转化成辐射场函数的输入输出的;

### 整体流程

1. 相机参数 + 像素网格 $\rightarrow$ 射线
2. 沿射线做 coarse 采样
3. 对采样点位置和方向做位置编码
4. coarse MLP 预测每个采样点的 $(\mathbf{c},\sigma)$
5. 根据 coarse 网络算出的权重分布，重点采样更可能有表面的区域
6. fine MLP 再预测一次 $(\mathbf{c},\sigma)$
7. 体渲染，得到每条射线的像素颜色
8. 所有像素拼成图像，训练时再和 GT 图像做损失

### 流程总图

对于整张图:

$$
(K,T_{c2w},H,W)
\rightarrow
\{(\mathbf{o}_{hw},\mathbf{d}_{hw})\}_{h,w}
\rightarrow
\{\mathbf{x}_{hw,i}\}
\rightarrow
\{(\mathbf{c}_{hw,i},\sigma_{hw,i})\}
\rightarrow
\hat I\in\mathbb{R}^{H\times W\times 3}
$$

对于单条射线:

$$
(K,T_{c2w},u,v)\rightarrow (\mathbf{o},\mathbf{d})
\rightarrow \{\mathbf{x}_i\}_{i=1}^{N}
\rightarrow \{(\mathbf{c}_i,\sigma_i)\}_{i=1}^{N}
\rightarrow \hat{\mathbf{C}}(\mathbf{r})\in\mathbb{R}^3
$$

---

## Step 1: 相机参数到射线
这一步是把图片像素和3d坐标的关系联系起来; 本质意思就是我们并不是无脑构建一个神经网络去端到端地拟合, 而是先植入一些cv的几何关系

### 输入

* 像素坐标 $(u,v)$，其中 $u\in[0,W-1], v\in[0,H-1]$
* 内参矩阵

$$
K=
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
\in\mathbb{R}^{3\times 3}
$$

* 外参 $T_c2w = [R | t]$: $(4,4)$
* 其中 `R`: $(3,3)$，`t`: $(3,)$

### 1. 像素到相机坐标系方向

先把像素反投影成相机坐标系中的方向向量：

$$
\tilde{\mathbf{d}}_{cam}=
\begin{bmatrix}
(u-c_x)/f_x\\
(v-c_y)/f_y\\
1
\end{bmatrix}
\in\mathbb{R}^3
$$



### 2. 相机坐标系到世界坐标系

世界坐标系下的射线方向：

$$
\mathbf{d}=R\mathbf{d}_{cam}\in\mathbb{R}^3
$$

射线原点就是相机中心：

$$
\mathbf{o}=\mathbf{t}\in\mathbb{R}^3
$$

于是单条射线写成：

$$
\mathbf{r}(t)=\mathbf{o}+t\mathbf{d},\quad t\ge 0
$$

然后我们会得到, 对整张图像:

* 射线原点 `rays_o`: $(H,W,3)$
* 射线方向 `rays_d`: $(H,W,3)$

射线原点是一个3d点; 射线方向是一个3d向量; 其中对与同一张图片来说, 射线原点是一样的; 

### 3. 采样

我们要考虑这里对每个像素都操作的话维度有点大, 所以一般我们会进行一些采样, 比如随机采取 $N_r$ 个像素;

* `rays_o_sampled`: $(N_r,3)$
* `rays_d_sampled`: $(N_r,3)$


### 输出

* `rays_o`: $(N_r,3)$
* `rays_d`: $(N_r,3)$

---

## Step 2: Coarse Sampling 粗糙采样

暂时先不考虑“粗糙”这个词, 就先把他当作一次正常的采样;

对每条射线，在 near 和 far 之间采样 $N_c$ 个3D点。

这里可以先把它理解成：

* $rays_o$ 决定射线从哪里出发
* $rays_d$ 决定射线朝哪个方向走
* $t$ 决定沿着这条射线走多远

所以这一阶段真正先采样的，不是 3D 点本身，而是一串深度标量 $z_vals$ 或 $t_vals$。  
然后再用

$$
\mathbf{x}=\mathbf{o}+t\mathbf{d}
$$

把这些深度标量变成真实的 3D 点。

### 输入

* `rays_o`: $(N_r,3)$
* `rays_d`: $(N_r,3)$
* near, far：标量, 一般设置为2,6
* coarse 采样数 $N_c$

### 1. 采样深度

常见是线性采样：

$$
t_i = t_{near} + \frac{i}{N_c-1}(t_{far}-t_{near}),\quad i=0,\dots,N_c-1
$$

对每条射线都得到一组深度：

$$
\mathbf{t}\in\mathbb{R}^{N_r\times N_c}
$$

这里的每个 $t_i$ 都是一个标量，表示：从相机中心出发，沿当前这条射线前进 $t_i$ 这么远。

比如：

* near = 2.0
* far = 6.0
* $N_c = 64$

那么这 64 个深度值大致就是把区间 $[2.0, 6.0]$ 划成 64 份：

$$
[2.0000,\ 2.0635,\ 2.1270,\ \dots,\ 5.9365,\ 6.0000]
$$

然后 进行stratified sampling; 也就是对每个采样深度加入一点点扰动,从而增加鲁棒性;


最后深度大概长这样:

$$
[2.0412,\ 2.1087,\ 2.1831,\dots]
$$

### 2. 深度到 3D 点

第 $i$ 个采样点：

$$
\mathbf{x}_i=\mathbf{o}+t_i\mathbf{d}
$$

批量写法：

$$
\mathbf{X}=\mathbf{O}+\mathbf{t}\odot \mathbf{D}
$$

其中:

* `O`: $(N_r,1,3)$
* `D`: $(N_r,1,3)$
* `t`: $(N_r,N_c,1)$
* `X`: $(N_r,N_c,3)$

### 输出

* `z_vals`: $(N_r,N_c,1)$ 
    代表某个射线 $(1-N_r)$ 的某个采样点 $(1-N_c)$ 的 深度(虽然这一步的目的是输出3d点的坐标, 但是这个深度信息还是需要保留的)
* `pts`: $(N_r,N_c,3)$ 
    一样,某个射线 $(1-N_r)$ 的某个采样点 $(1-N_c)$ 的 3d坐标

---

## Step 3: Positional Encoding 位置编码

NeRF 不直接把 $(\mathbf{x},\mathbf{d})$ 喂给 MLP进行学习，而是先做 Fourier feature / positional encoding。

### 输入

* 采样点位置 `pts`: $(N_r,N_c,3)$
* 被观测方向 `dirs`: $(N_r,3)$
    通常先扩展成 $(N_r,N_c,3)$; 对于同一个射线,每个采样点的被观察方向是一致的

这里为什么要编码?
因为原始的坐标值太"平滑"了, MLP 直接去拟合高频细节会比较困难;  
所以先把坐标映射到一组高频正余弦空间里从而提升对位置的感知

### 1. 编码公式

对一个标量 $p$, 编码函数 $\gamma(p)\in\mathbb{R}^{2L+1}$ 为:

$$
\gamma(p)=\Big[p,\sin(2^0p),\cos(2^0p),\dots,\sin(2^{L-1}p),\cos(2^{L-1}p)\Big]
$$

这里可以看出来:

* 原始标量 $p$ 自己占 $1$ 维
* 每个频率会对应一组 $\sin,\cos$，所以每个频率贡献 $2$ 维
* 一共有 $L$ 个频率

原始 NeRF 常见设置：

* 位置编码频率数 $L_x=10$
* 方向编码频率数 $L_d=4$

---

对 3D点位置向量 $\mathbf{x}=(x_1,x_2,x_3)$，逐维编码后拼接：

$$
\gamma(\mathbf{x})=
\Big[
\gamma(x_1),\gamma(x_2),\gamma(x_3)
\Big]
$$

如果把它完全展开:

$$
\gamma(\mathbf{x})=
\Big[
x_1,\sin(2^0x_1),\cos(2^0x_1),\dots,\sin(2^{L_x-1}x_1),\cos(2^{L_x-1}x_1),
$$

$$
x_2,\sin(2^0x_2),\cos(2^0x_2),\dots,\sin(2^{L_x-1}x_2),\cos(2^{L_x-1}x_2),
$$

$$
x_3,\sin(2^0x_3),\cos(2^0x_3),\dots,\sin(2^{L_x-1}x_3),\cos(2^{L_x-1}x_3)
\Big]
$$

---

同理方向编码：

$$
\gamma(\mathbf{d})=
\Big[
\gamma(d_1),\gamma(d_2),\gamma(d_3)
\Big]
$$

这里注意:

* 位置编码是对每个采样点都做一次
* 方向编码虽然一条射线只有一个方向, 但最后通常会复制到每个采样点上, 变成和 $pts$ 对齐的形状



所以维度通常为：

* $\gamma(x)$: $3 * (2 * 10 + 1) = 63$ 维
* $\gamma(d)$: $3 * (2 * 4 + 1) = 27$ 维

### 输出

* `pts`：$(N_r,N_c,C_x=63)$
   编码后采样点位置
* `dirs`：$(N_r,N_c,C_d=27)$
   编码后采样点被观测方向

---

## Step 4: Coarse MLP 粗糙预测辐射场

这一步就正式进入可学习的函数部分了;  
前面的几步都是几何前处理, 从这一步开始才是神经网络在拟合 3D 场景。
但是这一步是

### 输入

* 编码后位置`pts`：$(N_r,N_c,C_x)$，通常 $C_x=63$
* 编码后方向`pts`：$(N_r,N_c,C_d)$，通常 $C_d=27$

### 1. 辐射场函数

MLP 学习的是：

$$
F_\theta:(\gamma(\mathbf{x}),\gamma(\mathbf{d}))\longrightarrow (\mathbf{c},\sigma)
$$

更细一点，原始 NeRF 的结构是：

1. 先只用位置编码预测特征和密度
2. 再把特征和方向编码拼起来预测颜色

可以写成：

$$
(\mathbf{f},\sigma)=F_\theta^{(1)}(\gamma(\mathbf{x}))
$$

$$
\mathbf{c}=F_\theta^{(2)}(\mathbf{f},\gamma(\mathbf{d}))
$$

其中：

* $f$: $(C_f,)$，中间特征
* $\sigma$: 标量，密度
* $c$: $(3,)$，RGB

这里可以这样理解:

* $\sigma$ 回答的是: 这个位置有没有东西, 挡不挡光
* $c$ 回答的是: 如果看到这里, 它应该贡献什么颜色

对于一条射线上的每个采样点, coarse MLP 都会输出一次这样的结果。

### 输出

对 coarse 网络：

* `sigma_coarse`: $(N_r,N_c,1)$ 
* `rgb_coarse`: $(N_r,N_c,3)$

---

## Step 5: Coarse Volume Rendering 粗糙体渲染

一方面先给出一个粗糙的渲染结果, 
另一方面还负责告诉系统: 射线上哪些区域更值得继续细采样。

### 输入

* `rgb_coarse`: $(N_r,N_c,3)$
   粗糙采样的到的采样点的颜色
* `sigma_coarse`: $(N_r,N_c,1)$ 
   粗糙采样的到的采样点的透光率
* `z_vals_coarse`: $(N_r,N_c,1)$ 
   粗糙采样的到的采样点的深度(沿这条射线走多远)

### 1. 区间长度

$$
\delta_i = z_{i+1}-z_i
$$

通常最后一个区间补一个很大的值，表示射线延伸到无穷远。

### 2. 密度转 alpha

$$
\alpha_i = 1-\exp(-\sigma_i\delta_i)
$$

这里 $\alpha_i$ 可以理解成:

* 光线经过第 $i$ 个小区间时, 在这里"被吸收 / 被挡住"的概率

### 3. 透射率

$$
T_i=\prod_{j=1}^{i-1}(1-\alpha_j)
$$

这里 $T_i$ 表示:

* 光从相机出发, 在到达第 $i$ 个点之前, 一直都没有被前面挡住的概率

### 4. 每个采样点的权重

$$
w_i=T_i\alpha_i
$$

所以 $w_i$ 的意思就是:

* 前面没有被挡住
* 并且正好在这个位置被当前采样点贡献出来

这就是为什么体渲染最后本质上是一个加权求和。

### 5. 射线颜色

$$
\hat{\mathbf{C}}_{coarse}(\mathbf{r})=\sum_{i=1}^{N_c}w_i\mathbf{c}_i
$$
<!-- 
### 6. 深度图

$$
\hat D(\mathbf{r})=\sum_{i=1}^{N_c}w_i z_i
$$ -->

### 输出

* `weights_coarse`: $(N_r,N_c,1)$
   每条射线上每个采样点的权重: 
   它表示每条射线上的哪些采样点真正对最终像素贡献更大, 后面的 fine sampling 就是根据它来做的。
* `rgb_map_coarse`: $(N_r,3)$
   每条射线的最终体渲染颜色


---

## Step 6: Fine Sampling 精细采样

第二次的精细采样

不是重新均匀采样，而是把 coarse 权重当作概率密度，优先在高权重区域附近继续采样。

直觉上, 如果 coarse 网络已经告诉我们:

* 某一段几乎没有贡献, 那就没必要浪费太多采样点
* 某一段权重很高, 说明那里可能更接近真实表面, 那就应该多采一些

所以 fine sampling 的本质是:  
把有限的采样预算集中到更重要的位置。

注意,两次采样过程也是有参数的,最后也是通过训练来优化,要采样的范围;

### 输入

* `z_vals_coarse`: $(N_r,N_c,1)$ 
   粗糙采样点的深度
* `weights_coarse`: $(N_r,N_c,1)$
   粗糙采样点的权重(上一步中计算出来的)
*  $N_f$: int
   fine 采样的采样数

### 1. 构造 PDF / CDF

先把权重归一化为 PDF：

$$
p_i=\frac{w_i}{\sum_k w_k}
$$

再构造累计分布：

$$
\mathrm{CDF}_i=\sum_{j=1}^{i}p_j
$$

### 2. 逆变换采样

从均匀分布采样 $u\sim U[0,1]$，通过 CDF 反查得到新的采样深度 $\tilde z_j$。

得到 fine 采样点后，与 coarse 深度合并再排序：

$$
\mathbf{z}_{all}=\mathrm{sort}\Big(\mathbf{z}_{coarse}\cup \mathbf{z}_{fine}\Big)
$$

然后再像 Step 2 那样:

$$
\mathbf{x}_i=\mathbf{o}+z_i\mathbf{d}
$$

把这些新的深度转成新的 3D 采样点。

### 输出

假设总采样点数 $N=N_c+N_f$：

* `z_vals_fine`: $(N_r,N,1)$
* `pts_fine`: $(N_r,N,3)$

---
<!-- ########################################## -->
## Step 7: Fine MLP 精细预测辐射场

对合并后的所有采样点，再通过 fine 网络预测：

$$
F_\phi:(\gamma(\mathbf{x}),\gamma(\mathbf{d}))\longrightarrow (\mathbf{c},\sigma)
$$

通常 fine 网络结构和 coarse 网络相同，但参数不同。

这里本质上和 Step 4 是同一件事, 只是这一次采样点分布更合理了;  
coarse 是先粗略扫一遍, fine 是把更多计算资源放在疑似表面附近。

### 输入

* `pts_fine`：$(N_r,N,C_x)$
   编码后位置
* `dirs`：$(N_r,N,C_d)$
   编码后方向; 两次采样只是对点进行采样, 方向不用; 所以沿用之前的就行

### 输出

* `rgb_fine`: $(N_r,N,3)$
* `sigma_fine`: $(N_r,N,1)$ 

---
<!-- ########################################## -->
## Step 8: Fine Volume Rendering 体渲染得到最终像素颜色

再做一次体渲染：

$$
\alpha_i = 1-\exp(-\sigma_i\delta_i)
$$

$$
T_i=\prod_{j=1}^{i-1}(1-\alpha_j)
$$

$$
w_i=T_i\alpha_i
$$

$$
\hat{\mathbf{C}}_{fine}(\mathbf{r})=\sum_{i=1}^{N}w_i\mathbf{c}_i
$$

这一步和 Step 5 的公式本质一样, 只不过现在用的是更密集、更准确的采样结果;  
所以最后得到的 `rgb_map_fine` 才通常被当作最终输出。

### 输出

* 每条射线最终颜色：`rgb_map_fine`: $(N_r,3)$
* 整张图像重排后：`image`: $(H,W,3)$

---
<!-- ########################################## -->
## Step 9: 训练目标

训练时，有真实图像像素颜色 $\mathbf{C}(\mathbf{r})$，也有预测颜色 $\hat{\mathbf{C}}(\mathbf{r})$。

原始 NeRF 一般同时监督 coarse 和 fine：

$$
\mathcal{L} =
\sum_{\mathbf{r}\in\mathcal{R}}
\left\|
\hat{\mathbf{C}}_{coarse}(\mathbf{r})-\mathbf{C}(\mathbf{r})
\right\|_2^2
+
\sum_{\mathbf{r}\in\mathcal{R}}
\left\|
\hat{\mathbf{C}}_{fine}(\mathbf{r})-\mathbf{C}(\mathbf{r})
\right\|_2^2
$$

### 输入

* `rgb_map_coarse`: $(N_r,3)$
* `rgb_map_fine`: $(N_r,3)$
* `target_rgb`: $(N_r,3)$

### 输出

* 标量损失 `loss`: $(1,)$

这里也就是说:

* coarse 不能乱预测, 因为它自己也要被监督
* fine 负责最终更精细的结果

所以整个训练是 end-to-end 的;  
从最终 loss 反向传播时, coarse / fine / 位置编码前的所有可学习参数都会一起更新。

---

## 从“单个像素”到“整张图片”的函数链路

如果按推理时渲染整张图来理解，整个维度流可以记成：

1. 输入相机参数：
   $K: (3,3)$, $T_c2w: (4,4)$
2. 生成整张图像的射线：
   `rays_o` $(H,W,3)$，`rays_d` $(H,W,3)$
3. 展平成射线 batch：
   $(H\times W,3)$
4. 每条射线 coarse 采样 $N_c$ 个点：
   `pts_coarse` $(H\times W,N_c,3)$
5. coarse 网络输出：
   $rgb/sigma$ 分别是 $(H\times W,N_c,3)$ 和 $(H\times W,N_c)$
6. coarse 渲染：
   `rgb_map_coarse` $(H\times W,3)$
7. 层次采样后总点数变成 $N_c+N_f$
8. fine 网络输出：
   $(H\times W,N_c+N_f,3)$ 和 $(H\times W,N_c+N_f)$
9. fine 渲染：
   `rgb_map_fine` $(H\times W,3)$
10. reshape 回图像：
   `image` $(H,W,3)$
