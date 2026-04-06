# ViT

## Overview

ViT 的核心思想是:

* 不再把图像直接当二维卷积信号处理
* 而是把图像切成 patch, 再当作 token 序列输入 Transformer

因此它本质上是把 NLP 里的 Transformer 结构迁移到视觉任务中。

---

## Patchify

设输入图像:

$$
I\in\mathbb{R}^{H\times W\times C}
$$

把它切成大小为 $P\times P$ 的 patch 后, patch 数量为:

$$
N=\frac{HW}{P^2}
$$

每个 patch 被展平并线性投影成 token:

$$
z_i=E\cdot \mathrm{patch}_i
$$

于是图像被转成一个长度为 $N$ 的 token 序列。

---

## CLS Token 与位置编码

ViT 通常在 token 序列前面加一个额外的 `[CLS]` token:

$$
z_0
$$

它用于汇聚整张图的全局信息, 常用于分类头。

由于 Transformer 本身不带空间顺序信息, 因此还需要加入位置编码:

$$
\tilde z_i=z_i+e_i
$$

这样模型才能感知 patch 的空间位置。

---

## 多头注意力

ViT 的多头注意力和标准 Transformer 一样, 先计算:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

然后用缩放点积注意力:

$$
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

这里除以 $\sqrt{d_k}$ 的作用是:

* 防止 $QK^T$ 数值过大
* 避免 softmax 进入过饱和区域
* 保持梯度更稳定

与 GPT 的 causal attention 不同, ViT 的自注意力一般**不带 mask**。

---

## 总结

ViT 的整体流程可以概括成:

1. 图像切 patch
2. patch 线性投影成 token
3. 加入 `[CLS]` token 和位置编码
4. 送入多层 Transformer
5. 用 `[CLS]` 或 patch tokens 完成下游任务
