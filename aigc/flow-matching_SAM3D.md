# flow matching 案例: SAM 3D Objects

## Overview

Meta 的 `single-image to 3D` 生成模型:

$$
(I, M)\rightarrow \text{3D object}
$$

完整问题可以写成条件生成分布:

$$
q(S,T,R,t,s\mid I,M)
$$

其中:

* $I$: 输入图像
* $M$: 目标物体 mask
* $S$: 形状
* $T$: 纹理
* $R$: rotation
* $t$: translation
* $s$: scale

<figure><img src="/aigc/assets/image.png" alt="SAM 3D full architecture" width="1200"></figure>

---

这章只关注Course部分:

$$
(I,M)\rightarrow (O,R,t,s)
$$

其中:

* $O\in\mathbb{R}^{64\times 64\times 64}$: coarse voxel
* $R\in\mathbb{R}^6$: 6D rotation
* $t\in\mathbb{R}^3$: translation
* $s\in\mathbb{R}^3$: scale

关注:

* `Flow Matching` 生成范式
* `MoT` 作为`Flow Matching`速度场预测
* 图片与 mask 这类“prompt信息” 如何引入

<figure><img src="/aigc/assets/SAM3D_Course.png" alt="SAM 3D upper pipeline" width="1200"></figure>


## Flow Matching 生成框架

**数学形式**

设:

* $x_1$: 干净目标分布
* $x_0\sim\mathcal{N}(0,I)$: 初始高斯噪声分布
* $\tau\in[0,1]$: 虚拟生成时间; 也就是说整个总步长限定就是1, 如果`n`步走完, 每步的步长为`1/n`; SAM3D中训练的默认步长是25步
   这里的 $\tau\in[0,1]$ 只是 **单次生成轨迹上的时间**。所以“$\tau=1$”只表示: 这次采样从噪声走到了终点; 如果训练收敛了: 模型参数在训练过程中就学好了这个速度场

用一条直线插值路径:

$$
x_\tau = \tau x_1 + (1-\tau)x_0
$$

因为这条路径是直线, 所以真实速度场就是:

$$
v = \dot{x}_\tau = x_1 - x_0
$$

然后训练一个**可微模型**去拟合这个速度生成网络:

$$
v_\theta(x_\tau, c, \tau)
$$

其中:

* $x_\tau$: 当前状态
* $c$: 条件信息
* $\tau$: 当前时刻

loss 最简单可以理解成:

$$
L=
\mathbb{E}_{\tau,x_0}
\left[
\|v-v_\theta(x_\tau,c,\tau)\|^2
\right]
$$

**训练时**

1. 取真实目标 $x_1$
2. 采样高斯噪声 $x_0$
3. 随机采样时间 $\tau$
4. 构造当前状态 $x_\tau$
5. 把 $(x_\tau,c,\tau)$ 喂给网络
6. 网络输出预测速度 $v_\theta$
7. 和真实速度 $x_1-x_0$ 做 MSE
8. 反向传播更新网络参数

**推理时**

1. 只有噪声初值 $x_0$
2. 从 $\tau=0$ 开始
3. 网络输出当前速度 $v_\theta(x_\tau,c,\tau)$
4. 用 ODE solver 更新:

$$
x_{\tau+\Delta\tau}=x_\tau+\Delta\tau\cdot v_\theta(x_\tau,c,\tau)
$$

5. 不断重复, 直到 $\tau=1$

**可微模型$v_\theta(x_\tau,c,\tau)$的需求**

从数学上说, Flow Matching 只要求有一个模型:

$$
v_\theta(x_\tau,c,\tau)
$$

满足:

* 输入当前状态
* 输入条件
* 输出与当前状态同维度的速度

所以理论上可以是:

* MLP
* CNN / U-Net
* Transformer

SAM 3D 之所以选 Transformer 是因为要同时处理:

* 大量 3D latent token
* 图像条件 token
* shape / layout 两种不同模态


## SAM 3D中的flow matching

现在把上面的通用框架落到 `Geometry Model` 里

这一阶段要生成的是:

$$
q(O,R,t,s\mid I,M)
$$

然后我们每步进行优化的token如下:(也就是最终输出)

* shape token: 
   shape token 是 64 * 64 * 64 * 1 的维度, 也就是把一块空间划分成 64 * 64 * 64 的小块(体素), 然后每一块里面是一个1维向量; 
* layout token
   layout token 是 1*12 维度向量, 代表物体位姿, 包含 $R\in\mathbb{R}^6$, $t\in\mathbb{R}^3$, $s\in\mathbb{R}^3$ 

--- 
**初始化**
这些token初始的时候就是标准高斯采样得到的噪声;
 
---
**token投影网络**
这一步的目的是下采样投影到潜空间, 减少一些计算开销

**shape token 的状态变量**

首先不直接在 
$$
z_{shape}\in\mathbb{R}^{64\times 64\times 64\times 1}
$$

体素空间里作为速度场预测模型的直接输入, 而是先把 shape 映射到一个更紧凑的 latent 空间:

$$
z_{shape}\in\mathbb{R}^{16\times 16\times 16\times 8}
$$

含义是:

* 空间上从 `64 -> 16` 做了压缩
* 每个空间块用一个 `8` 维 latent feature 表示

所以展平之后, shape 对应:$ 16\times 16\times 16=4096 $ 个 token


**layout token 的状态变量**

layout 不需要一个 3D 网格, 因为它维度很小, 所以直接形成:`1` 个 layout token

总共加起来是 4096个shape token + 1个 layout token

---
**Mixture-of-Transformers**
在 Geometry Model 里,   真正被 Flow Matching loss 直接训练的主干就是:

* `1.2B` 参数的 `Flow Transformer`; 具体结构用的是 `Mixture-of-Transformers (MoT)`

也就是速度场预测网络:

$$
v_\theta(x_\tau,c,\tau)
$$

这个图的流程是从上往下流; 

<figure><img src="/aigc/assets/MoT1.png" alt="SAM 3D upper pipeline" width="1200"></figure>



**然后我们来看输入; 输入包括两类 shape token 和 layout token**


进入 MoT 的 shape token 是:

$$
4096 \text{ tokens},\ each\in\mathbb{R}^{1024}
$$

进入 MoT 的 layout token 是:

$$
1 \text{ tokens},\ each\in\mathbb{R}^{1024}
$$


---
**shape token 和 layout token的自注意力计算**

**拼接进场 (Concatenation)**

系统会把 4096 个 Shape Tokens 和 1 个 Layout Token 直接在序列维度上拼接起来，排成一列。现在我们得到了一个长度为 4097 的总token序列。

**独立的 QKV 映射 (Independent Projection)**
它们虽然坐在同一桌，但QKV是分开的：
- Shape Tokens 使用专属于 Shape 的线性层权重（$W_Q, W_K, W_V$），计算出自己的 $Q_{shape}, K_{shape}, V_{shape}$

-  Layout Token 使用专属于 Layout 的线性层权重，计算出自己的 $Q_{layout}, K_{layout}, V_{layout}$

算完之后，系统再把它们拼回去，形成一个长度为 4097 的 $Q_{total}$、$K_{total}$ 和 $V_{total}$。

<figure><img src="/aigc/assets/Mutil-Modal_Self-Attention1.png" alt="Multi-modal self-attention" width="1100"></figure>

**注意力矩阵**
4 个象限的注意力矩阵与 Mask接下来进行标准的自注意力计算：用 $Q_{total}$ 乘以 $K_{total}$ 的转置。这就生成了一个 $4097 \times 4097$ 的巨大注意力打分矩阵。

这个矩阵天然被划分成了 4 个象限（这正是论文 Figure 2 中画的那个带灰色的正方形方块）：

象限一（左上角）：Shape Q $\times$ Shape K意义：Shape Token 互相观察，决定长成什么形状。操作：全开放，正常前向传播，正常更新梯度。

象限二（右下角）：Layout Q $\times$ Layout K意义：Layout Token 自己内部消化（虽然这里只有 1 个 Token）。操作：全开放，正常前向传播，正常更新梯度。

象限三（右上角）：Shape Q $\times$ Layout K意义：决定物体的形状时，要不要看它的旋转位姿？操作：通**常完全 Mask 掉（强行设为 0）**。因为杯子无论怎么转，它的物理形状都是杯子，形状生成不应该被位姿干扰。

象限四（左下角）：Layout Q $\times$ Shape K （Stop Grad 的核心区域）意义：Layout Token 在预测“怎么摆放”时，去观察“我到底长什么样”。操作：在前向传播时，它是畅通的，Layout 会把 Shape 的特征（Value）加权拿过来参考。但是，系统在这个象限铺设了一层“单向玻璃 (Stop Gradient Mask)”。

在反向传播计算损失时，如果发现位姿摆错了，系统在沿着梯度往回传播时，走到这里会掐断。
这就保证了：Layout 可以“使用” Shape 的信息，但 Layout 反馈更新梯度不让 Shape 更新权重。

---
**提示图片信息融合进MoT**

<figure><img src="/aigc/assets/MoT1.png" alt="SAM 3D upper pipeline" width="1200"></figure>
回到图片, 经过FFN之后就到交叉注意力了; 这里的交叉注意力是和我们给的提示图片进行注意力计算;

<figure><img src="/aigc/assets/cross_attention.png" alt="SAM 3D upper pipeline" width="1200"></figure>

如图, 紫色箭头就是我们的shape token 和 layout token的流动过程;

红色框就是我们的提示图片“prompt”引入的部分(对于我们用户, 这个提示图片就是我们的输入; 但是对于生成模型 我们的输入指示作为生成过程的引导)

首先，输入的图像 (Image) 和掩码 (Object Mask) 会通过作为图像编码器的 DINOv2 。
这个过程并不仅仅是提取一张图，而是提取了两对图像，最终产生 4 组条件特征 (Conditioning tokens) ：
- 局部裁剪视角 (Cropped object)：原图被 Mask 裁剪后的局部图像特征及其对应的二值掩码，用于提供高分辨率的局部细节 。
- 全局完整视角 (Full image)：完整的原图特征及其全局二值掩码，用于提供全局的场景上下文信息 

在提取出这些富含视觉和语义信息的条件特征 Token 后，它们被直接送入 Mixture of Transformers (MoT) 中

---
**[shape token 和 layout token] 与 [提示图片特征 token] 交叉注意力计算**

当那 4097 个 Tokens（4096 个 Shape + 1 个 Layout）刚刚在“多模态自注意力层”里计算完成后（四象限 Mask 注意力）之后，它们马上就会进入下一步——与 DINOv2 提取的图片特征进行融合。

这里使用的**基本理论**是：**交叉注意力机制 (Cross-Attention Mechanism)**。可以拆解为以下四个精确的步骤：

- **在交叉注意力中，Q、K、V 的来源被严格拆分了，不再是自己和自己算：**

   - **Query (查询, Q)：** 来源于刚刚经过自注意力层处理完毕的 **4097 个 3D 潜变量 Tokens**

   - **Key (键, K) 和 Value (值, V)：** 来源于 **DINOv2 提取出的图像特征 (Conditioning tokens)**
   DINOv2 把局部裁剪图和全局原图切成了一个个的图像块 (Patches)，提取出了富含高级语义的视觉特征序列

- **跨模态打分 (计算 $Q \times K^T$)**

   - 接下来，系统拿着 3D 侧的 $Q$，去和 2D 图像侧的 $K$ 进行矩阵乘法运算。
   - **物理意义：** 这其实是在计算**相似度或相关性**。那 4097 个 3D Tokens 正在逐一扫视 DINOv2 提取出的所有图像特征块
   - **结果：** 这一步会得出一个“注意力权重矩阵”。比如，负责生成“杯口”的 3D Token，在和图像中“杯口”位置的图像特征计算点积时，会得到一个极高的分数；而和图像背景部分的得分则接近于 0

- **提取视觉信息 (乘以 V)**
   - 有了这个打分矩阵，系统就会用它去对图像的 Value ($V$) 进行加权求和。
   - **物理意义：** “各取所需”。如果那个 3D Token 极其关注图片右上角的特征，它就会把右上角特征的 $V$（比如包含了“红色”、“反光”等高维语义信息）大量提取过来。
   - **结果：** 此时，我们得到了一个提取了外部视觉指导信息的**全新特征张量**，它的维度和最初的 4097 个 3D Tokens 完全一致。

- **融合与残差连接 (Residual Connection)**
   - 在标准的 Transformer 块 (Block) 中，交叉注意力层输出的这个包含了图像信息的特征，并不会直接替换掉原来的 3D Tokens，而是通过**残差连接 (Add & Norm)** 的方式加进去：

$$Tokens_{n+1} = LayerNorm(Tokens_{n} + CrossAttention(Tokens_{n}, Image_{K}, Image_{V}))$$


---
**输出速度场**


然后再通过 output projection 回到各自真正的状态空间:

* shape: `1024 -> 8`
* layout: `1024 -> 12`

---
**token 更新**

拿到这个速度场之后, 外部的 ODE solver 就会按步长去更新当前 token:

$$
x_{\tau+\Delta\tau}=x_\tau+\Delta\tau\cdot v_\theta(x_\tau,c,\tau)
$$

也就是说:

* 当前的 shape token 按 shape velocity 更新
* 当前的 layout token 按 layout velocity 更新

更新完之后再重新送回 `MoT`, 再做下一步;

这样经过 `n` 步之后, token 会逐渐从噪声收敛到稳定的 shape latent 和 layout latent;

最后:

* shape latent 再 project / decode 回原本的体素维度, 得到 `64 x 64 x 64 x 1` 的 coarse voxel
* layout latent 则回到 `1 x 12`, 也就是最终的 $(R,t,s)$
