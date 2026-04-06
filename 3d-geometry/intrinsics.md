# Intrinsics

## Overview

相机内参描述的是:

* 相机坐标系中的点如何投影到像素平面
* 归一化成像平面与像素坐标之间的线性关系
* 焦距、主点和像素坐标轴几何关系

标准针孔相机模型下, 内参矩阵写成:

$$
K=
\begin{bmatrix}
f_x & s & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
$$

其中:

* $f_x, f_y$: x / y 方向的等效焦距, 单位通常是 pixel
* $c_x, c_y$: 主点坐标
* $s$: skew, 描述像素坐标轴是否正交

严格来说, 线性内参矩阵 $K$ 有 5 个自由度。  
但在大多数实际相机里, 常近似认为像素坐标轴正交, 因此:

$$
s \approx 0
$$

这时常把内参简化成 4 个参数:

$$
(f_x,f_y,c_x,c_y)
$$

---

## 参数含义

### 1. $f_x, f_y$

$f_x, f_y$ 控制的是:

* 像素坐标对归一化坐标的缩放
* 视场角大小
* 投影后物体在图像中的“放大程度”

它们本质上是焦距在像素单位下的表示。  
若物理焦距为 $f$、像素尺寸分别为 $d_x,d_y$, 则可理解为:

$$
f_x \approx \frac{f}{d_x},\qquad
f_y \approx \frac{f}{d_y}
$$

直观上:

* $f_x,f_y$ 越大: 视场角越小, 画面越“放大”
* $f_x,f_y$ 越小: 视场角越大, 画面越“广角”

若图像宽高分别为 $W,H$, 则常见近似关系为:

$$
f_x=\frac{W}{2\tan(\mathrm{FOV}_x/2)},\qquad
f_y=\frac{H}{2\tan(\mathrm{FOV}_y/2)}
$$

### 2. $c_x, c_y$

$c_x, c_y$ 表示主点, 即光轴与成像平面的交点在像素坐标中的位置。

它们的作用是:

* 决定投影中心在图像中的偏移
* 把归一化坐标平移到真实像素坐标系

很多情况下主点接近图像中心, 但不一定严格等于:

$$
\left(\frac{W}{2},\frac{H}{2}\right)
$$

### 3. $s$

$s$ 是 skew, 用于描述:

* 像素坐标系的 x / y 轴是否正交

现代数字相机里通常取:

$$
s=0
$$

因此很多视觉任务默认忽略该项。

---

## 畸变参数

真实镜头通常还包含非线性畸变, 因而实际标定文件中的“内参”往往不止矩阵 $K$。

常见的 OpenCV 畸变模型包含:

* $k_1,k_2,k_3$: 径向畸变
* $p_1,p_2$: 切向畸变

因此如果把线性内参和畸变一起算进去, 参数维度通常会增加到 9 到 13 维左右。

需要区分:

* 线性内参: 矩阵 $K$
* 非线性畸变: 额外的 distortion coefficients

---

## K 的前向作用

### 从相机坐标到归一化平面

设相机坐标系中的一点为:

$$
\mathbf{X}_c=
\begin{bmatrix}
X_c\\
Y_c\\
Z_c
\end{bmatrix}
$$

则其归一化平面坐标为:

$$
x_n=\frac{X_c}{Z_c},\qquad
y_n=\frac{Y_c}{Z_c}
$$

写成齐次形式:

$$
\begin{bmatrix}
x_n\\
y_n\\
1
\end{bmatrix}
=
\frac{1}{Z_c}
\begin{bmatrix}
X_c\\
Y_c\\
Z_c
\end{bmatrix}
$$

### 从归一化平面到像素坐标

内参矩阵 $K$ 的核心作用就是:

$$
\lambda
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
K
\begin{bmatrix}
X_c\\
Y_c\\
Z_c
\end{bmatrix}
$$

其中:

$$
\lambda = Z_c
$$

等价地, 也常写成:

$$
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
K
\begin{bmatrix}
x_n\\
y_n\\
1
\end{bmatrix}
$$

展开后得到:

$$
u=f_xx_n+sy_n+c_x
$$

$$
v=f_yy_n+c_y
$$

若忽略 skew, 则:

$$
u=f_x\frac{X_c}{Z_c}+c_x,\qquad
v=f_y\frac{Y_c}{Z_c}+c_y
$$

这说明:

* $f_x,f_y$ 负责缩放
* $c_x,c_y$ 负责平移
* $K$ 把“几何坐标”变成“像素坐标”

---

## K 的逆作用

### 从像素坐标回到归一化坐标

已知像素坐标 $(u,v)$, 可以先通过 $K^{-1}$ 回到归一化相机平面:

$$
\begin{bmatrix}
x_n\\
y_n\\
1
\end{bmatrix}
=
K^{-1}
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
$$

若 $s=0$, 则公式最常写成:

$$
x_n=\frac{u-c_x}{f_x},\qquad
y_n=\frac{v-c_y}{f_y}
$$

### 从像素坐标生成相机射线

这一步在 NeRF、SLAM、SfM、深度估计里非常常见。  
给定像素 $(u,v)$, 相机坐标系中的射线方向可写成:

$$
\mathbf{d}_{cam}=
\begin{bmatrix}
(u-c_x)/f_x\\
(v-c_y)/f_y\\
1
\end{bmatrix}
$$

或者写成:

$$
\mathbf{d}_{cam}\propto
K^{-1}
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
$$

随后若已知相机外参 $R,t$, 就可以把这条射线变换到世界坐标系。

---

## 与外参一起使用

如果场景点在世界坐标系中写作:

$$
\mathbf{X}_w=
\begin{bmatrix}
X_w\\
Y_w\\
Z_w\\
1
\end{bmatrix}
$$

则完整投影关系为:

$$
\lambda
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
K
\begin{bmatrix}
R & t
\end{bmatrix}
\mathbf{X}_w
$$

这条公式可以理解成三步:

1. 外参把世界点变到相机坐标系
2. 透视除法得到归一化平面坐标
3. 内参矩阵 $K$ 把归一化坐标变到像素坐标

因此:

* 外参回答“相机在哪里、朝哪里看”
* 内参回答“相机内部如何成像”

---

## 改变图像尺寸时如何修改 K

内参矩阵不是永远不变的。  
只要图像坐标系发生变化, $K$ 往往也要跟着改。

### 1. 图像缩放

若图像在 x / y 方向分别缩放为 $\alpha_x,\alpha_y$, 则新的内参通常写成:

$$
f_x'=\alpha_x f_x,\qquad
f_y'=\alpha_y f_y
$$

$$
c_x'=\alpha_x c_x,\qquad
c_y'=\alpha_y c_y
$$

### 2. 图像裁剪

若从原图左上角裁掉偏移 $(\Delta u,\Delta v)$, 则:

$$
c_x'=c_x-\Delta u,\qquad
c_y'=c_y-\Delta v
$$

而焦距通常不变:

$$
f_x'=f_x,\qquad f_y'=f_y
$$

这在数据预处理、resize、crop、multi-scale 训练里都很常见。

---

## 常见用法总结

内参矩阵 $K$ 最常见的用途有三类:

### 1. 3D 点投影到图像

已知相机坐标系点 $\mathbf{X}_c$, 求像素坐标 $(u,v)$。

### 2. 像素反投影成射线

已知像素 $(u,v)$, 用 $K^{-1}$ 恢复归一化方向, 再生成 3D 射线。

### 3. 与外参联合建模成像过程

用

$$
\lambda \mathbf{u}=K[R|t]\mathbf{X}_w
$$

统一表示从世界点到像素坐标的完整映射。
