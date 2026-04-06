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


## K 的前向作用




设相机坐标系中的一点为:

$$
\mathbf{P}=
\begin{bmatrix}
X\\
Y\\
Z
\end{bmatrix}
$$

这一点在图像上的坐标为:
$$
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
$$

内参矩阵 $K$ 就是:

$$
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}=
K
\begin{bmatrix}
X/Z \\
Y/Z \\
1
\end{bmatrix}
$$


**从相机坐标到归一化平面**
首先从3d点坐标 $P$ 出发

则其归一化平面坐标 $p$ 为:

$$
x=\frac{X}{Z},\qquad
y=\frac{Y}{Z}
$$

写成齐次形式:

$$
p = 
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}=
\frac{1}{Z}
\begin{bmatrix}
X\\
Y\\
Z
\end{bmatrix} = 
\begin{bmatrix}
X/Z\\
Y/Z\\
1
\end{bmatrix} 
$$

**从归一化平面到像素坐标**

然后我们从这个归一化坐标 $p$ 继续;

$$
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}=
K
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}
$$


则:

$$
u=f_x x+c_x,\qquad
v=f_y y+c_y
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
x\\
y\\
1
\end{bmatrix} =
K^{-1}
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
$$

公式最常写成:

$$
x_n=\frac{u-c_x}{f_x},\qquad
y_n=\frac{v-c_y}{f_y}
$$


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
\end{bmatrix}=
K
\begin{bmatrix}
R & t
\end{bmatrix}
\mathbf{X}_w
$$

这条公式可以理解成三步:

1. 外参把世界点变到相机坐标系($R$ 旋转 $t$ 平易)
2. 透视除法得到归一化平面坐标
3. 内参矩阵 $K$ 把归一化坐标变到像素坐标

