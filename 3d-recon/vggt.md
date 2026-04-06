# VGGT

## Overview

从当前 PDF 内容看, VGGT 属于前馈式几何重建路线中的代表方法之一。  
它的重要特点是:

* 不依赖传统逐场景优化
* 直接前向预测几何结果
* 输出中通常包含 3D 点图、位姿和置信度

在长序列场景中, VGGT 还会把视频切成多个带重叠的 chunk,  
先做局部重建, 再做 chunk 之间的对齐。

---

## 局部输出

对每个 chunk, VGGT 一般会输出:

* 相机位姿
* 稠密 3D 点图
* 每个像素点的置信度图

因此一个 chunk 的几何结果可以理解成:

$$
(P_k,c_k,T_k)
$$

其中:

* $P_k$: 点图
* $c_k$: 置信度
* $T_k$: chunk 内部自洽的位姿

---

## 长序列中的问题

不同 chunk 虽然内部自洽, 但彼此之间通常不共享同一全局坐标系。  
因此 chunk 之间往往存在:

* 尺度漂移
* 旋转偏差
* 平移偏差

这也是为什么长视频场景里需要:

* Sim(3) 对齐
* IRLS + Umeyama
* 甚至更一般的 SL(4) 对齐

---

## 与传统方法的区别

VGGT 这类前馈式几何模型和传统点云配准方法的主要区别在于:

* 传统方法输入是两组点云, 往往还要额外做特征匹配或最近邻搜索
* 前馈式几何模型通常直接输出带 confidence 的对应几何

因此后续对齐阶段可以利用:

* 天然重叠帧
* 更稳定的对应关系
* 模型置信度

来做更鲁棒的配准。

---

## 相关主题

当前笔记体系里, VGGT 相关的几何后处理主要见:

* [IRLS + Umeyama](../alignment/irls_umeyama.md)
* [SL(4) Point Alignment](../alignment/sl4_alignment.md)
* [Bundle Adjustment](../optimization/bundle_adjustment.md)
