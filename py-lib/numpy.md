# NumPy 常做什么

NumPy 是 Python 里最基础的数值计算库，常用来表示向量、矩阵、多维数组，以及做切片、运算、统计和线性代数

```python
import numpy as np
```

```bash
pip install numpy
```

## 创建数组

最基础的用法就是先创建一个 `ndarray`

```python
import numpy as np

a = np.array([1, 2, 3]) # 从 Python 列表创建一维数组
b = np.array([[1, 2], [3, 4]]) # 创建二维数组
c = np.zeros((2, 3)) # 创建一个 2x3 的全 0 数组
d = np.ones((2, 3)) # 创建一个 2x3 的全 1 数组
e = np.arange(0, 10, 2) # 创建 0 到 10 之间步长为 2 的数组
f = np.linspace(0, 1, 5) # 在 0 到 1 之间均匀取 5 个点
```

常见例子：

- 构造一个向量或矩阵
- 初始化权重、mask、坐标
- 生成一段连续数值

## 看数组的形状和类型

写 NumPy 时，经常第一眼就先看 `shape` 和 `dtype`

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32) # 创建一个 2x3 的 float32 数组
print(arr.shape) # 查看数组形状，结果通常是 (2, 3)
print(arr.ndim) # 查看数组维度个数，二维数组这里会输出 2
print(arr.dtype) # 查看数组里元素的数据类型
print(arr.size) # 查看数组总共有多少个元素
```

常见例子：

- 检查数据是不是你想要的维度
- 看图像是 `(H, W)` 还是 `(H, W, 3)`
- 排查类型不一致导致的报错

## 索引和切片

NumPy 最常见的操作之一就是取出一部分数据

```python
import numpy as np

arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]) # 创建一个 3x3 数组
print(arr[0, 1]) # 取第 0 行第 1 列，结果是 20
print(arr[1]) # 取第 1 行
print(arr[:, 0]) # 取第 0 列
print(arr[0:2, 1:3]) # 取前两行、后两列组成的子数组
```

常见例子：

- 取一张图像里的局部区域
- 取一个 batch 里的前几个样本
- 取某一列特征

## 改形状、转置、展平

很多时候数据本身没变，只是换一个排布方式

```python
import numpy as np

arr = np.arange(12) # 创建一个包含 0 到 11 的一维数组
b = arr.reshape(3, 4) # 把一维数组改成 3x4
c = b.T # 对二维数组做转置，行列交换
d = b.flatten() # 把二维数组重新拉平成一维数组
```

常见例子：

- 把图像拉平成特征向量
- 调整 batch 维和通道维
- 为矩阵运算准备合适的形状

## 数组运算

NumPy 支持逐元素运算，这点和 Python list 很不一样

```python
import numpy as np

a = np.array([1, 2, 3]) # 创建第一个数组
b = np.array([4, 5, 6]) # 创建第二个数组
print(a + b) # 逐元素相加，结果是 [5 7 9]
print(a - b) # 逐元素相减
print(a * b) # 逐元素相乘
print(a / b) # 逐元素相除
print(a ** 2) # 每个元素分别平方
```

常见例子：

- 向量加减
- 对整张图像做缩放
- 对一批数据统一归一化

## 广播

广播可以理解成：形状不同的数组，在规则允许时自动“补齐”后再运算

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]]) # 创建一个 2x3 数组
v = np.array([10, 20, 30]) # 创建一个长度为 3 的一维数组
result = arr + v # 按行广播，把 v 加到 arr 的每一行上
print(result) # 输出广播后的结果
```

常见例子：

- 给每一行加上同一个偏置
- 给 RGB 三个通道减均值
- 做批量归一化

## 条件筛选

NumPy 可以按条件取值，这在数据清洗里特别常见

```python
import numpy as np

arr = np.array([1, 5, 2, 8, 3, 9]) # 创建一个一维数组
mask = arr > 4 # 得到一个布尔数组，表示哪些元素大于 4
filtered = arr[mask] # 用布尔数组筛选出满足条件的元素
print(mask) # 输出 True / False 掩码
print(filtered) # 输出筛选后的结果
```

常见例子：

- 取出大于阈值的像素
- 删除无效值
- 根据条件筛选样本

## 聚合和统计

很多时候不是要保留所有元素，而是求和、均值、最大值这些统计量

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]]) # 创建一个 2x3 数组
print(arr.sum()) # 计算所有元素的总和
print(arr.mean()) # 计算所有元素的平均值
print(arr.max()) # 取所有元素里的最大值
print(arr.min()) # 取所有元素里的最小值
print(arr.mean(axis=0)) # 按列求均值
print(arr.mean(axis=1)) # 按行求均值
```

常见例子：

- 算数据均值和标准差
- 求每一列特征的统计量
- 求一张图像的平均亮度

## 拼接和堆叠

NumPy 也很常用来把多个数组拼在一起

```python
import numpy as np

a = np.array([[1, 2], [3, 4]]) # 创建第一个 2x2 数组
b = np.array([[5, 6], [7, 8]]) # 创建第二个 2x2 数组
c = np.concatenate([a, b], axis=0) # 按行拼接，结果是 4x2
d = np.concatenate([a, b], axis=1) # 按列拼接，结果是 2x4
e = np.stack([a, b], axis=0) # 新增一个维度后堆叠，结果是 2x2x2
```

常见例子：

- 合并多个样本
- 拼接特征
- 给数据增加 batch 维

## 随机数

随机初始化、随机采样也经常会用到 NumPy

```python
import numpy as np

np.random.seed(0) # 固定随机种子，方便复现结果
a = np.random.rand(2, 3) # 生成 2x3 的 0 到 1 之间均匀分布随机数
b = np.random.randn(2, 3) # 生成 2x3 的标准正态分布随机数
idx = np.random.randint(0, 10, size=5) # 随机生成 5 个 0 到 9 之间的整数
```

常见例子：

- 初始化参数
- 随机打乱样本
- 生成测试数据

## 矩阵乘法

NumPy 可以做线性代数运算，矩阵乘法是最常见的一个

```python
import numpy as np

A = np.array([[1, 2], [3, 4]]) # 创建第一个矩阵
B = np.array([[5, 6], [7, 8]]) # 创建第二个矩阵
C = A @ B # 做矩阵乘法
D = np.dot(A, B) # 用 dot 也可以做矩阵乘法
```

常见例子：

- 坐标变换
- 线性回归里的矩阵计算
- 神经网络里的线性层

## 常见 NumPy 流程

很多任务不是只用一个函数，而是一串操作连起来

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 100]], dtype=np.float32) # 创建一个二维数组
arr = arr / 255.0 # 先做归一化
mean = arr.mean(axis=0) # 按列计算均值
mask = mean > 0.01 # 用条件得到一个布尔掩码
result = mean[mask] # 取出满足条件的列均值
print(result) # 输出最后结果
```

