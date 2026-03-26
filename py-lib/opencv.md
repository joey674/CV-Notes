# OpenCV 常做什么

OpenCV 是一个计算机视觉工具库，常用来做图像读取、预处理、特征提取、视频处理;
默认的数据结构时numpy

```python
import cv2
import numpy as np
```

```bash
pip install opencv-python
```

## 读图、写图、显示图片

最基础的用途就是把图片读进来，处理后再保存。

```python
import cv2

img = cv2.imread("cat.jpg") # 从磁盘读取图片，返回一个 BGR 格式的数组
cv2.imshow("image", img) # 打开一个名为 image 的窗口显示图片
cv2.imwrite("cat_copy.jpg", img) # 把图片保存到新文件
cv2.destroyAllWindows() # 关闭所有 OpenCV 创建的窗口
```

## 图像缩放、裁剪、旋转

这类操作常用于预处理，让输入尺寸统一，或者只保留感兴趣区域。

```python
import cv2

img = cv2.imread("cat.jpg")
resized = cv2.resize(img, (224, 224)) # 把图片缩放到 224x224
crop = img[50:250, 100:300] # 按 [y1:y2, x1:x2] 裁剪一个局部区域
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 将图片顺时针旋转 90 度
```

## 颜色空间转换

OpenCV 默认读进来是 BGR，不是 RGB。很多任务会先转成灰度图、HSV 等颜色空间。

```python
import cv2

img = cv2.imread("cat.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 把 BGR 图转成灰度图
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 把 BGR 图转成 HSV 图
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 把 BGR 图转成 RGB 图，方便给 matplotlib 显示
```

## 阈值分割、边缘检测

这类方法常用来把前景和背景分开，或者找出图像轮廓。

```python
import cv2

img = cv2.imread("doc.jpg", 0) # 以灰度模式读取图片，0 表示直接读成单通道灰度图
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # 大于 127 的像素设为 255，小于等于 127 的设为 0
edges = cv2.Canny(img, 100, 200) # 用 Canny 算法提取边缘，100 和 200 是双阈值
```

常见例子：

- 文档扫描时把文字区域二值化
- 工业检测里提取零件边缘
- 车道线检测前先做边缘提取

## 滤波、去噪

当图像里有噪声时，通常会先做平滑处理。

```python
import cv2

img = cv2.imread("noisy.jpg")
blur1 = cv2.GaussianBlur(img, (5, 5), 0) # 用 5x5 高斯核做平滑，0 表示 sigma 自动计算
blur2 = cv2.medianBlur(img, 5) # 用中值滤波去噪，5 表示滤波窗口大小
```

常见例子：

- 拍照噪点较多时先平滑再做检测
- 二值化之前先做高斯滤波，让结果更稳定
- 中值滤波常用于去除椒盐噪声

## 特征点检测

OpenCV 可以提取图像中的关键点和描述子，用于匹配、拼接、定位等任务。

```python
import cv2

img = cv2.imread("building.jpg", 0) # 读取灰度图，特征点提取常直接在灰度图上做
orb = cv2.ORB_create() # 创建 ORB 特征提取器
keypoints, descriptors = orb.detectAndCompute(img, None) # 检测关键点，并计算每个关键点的描述子
out = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0)) # 把关键点画到图上方便观察
cv2.imwrite("orb_points.jpg", out) # 保存画好关键点的结果图
```

常见例子：

- 两张图片做匹配
- 全景图拼接
- SLAM / VO 中提取角点或局部特征

## 视频读取

OpenCV 也常用来逐帧读取视频，再对每一帧做处理。

```python
import cv2

cap = cv2.VideoCapture("demo.mp4") # 打开一个视频文件，也可以传 0 表示打开默认摄像头

while True:
    ret, frame = cap.read() # ret 表示是否读成功，frame 是当前帧图像
    if not ret: # 如果没有读到帧，通常说明视频结束了
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 把当前帧转成灰度图
    cv2.imshow("image", gray) # 显示当前处理后的帧

cap.release() # 释放视频对象或摄像头资源
cv2.destroyAllWindows() # 关闭所有窗口
```

## 相机标定

相机标定的目标是求相机内参、畸变系数，用于更准确的测量和重建。

```python
import cv2
import numpy as np

gray = cv2.imread("chessboard.jpg", 0) # 读取一张棋盘格图片并转成灰度图
objpoints = [] # 用来保存世界坐标系中的角点
imgpoints = [] # 用来保存图像坐标系中的角点

objp = np.zeros((9 * 6, 3), np.float32) # 创建 9x6 棋盘格的三维点，先全部置零
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) # 给每个角点填入平面上的 x,y 坐标

ret, corners = cv2.findChessboardCorners(gray, (9, 6), None) # 在图里找 9x6 的棋盘格角点
if ret: # 如果成功找到角点
    objpoints.append(objp) # 记录这张图对应的世界坐标角点
    imgpoints.append(corners) # 记录这张图检测到的图像角点

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera( # 根据角点求相机参数
    objpoints, imgpoints, gray.shape[::-1], None, None # gray.shape[::-1] 是图像宽高
)
```

常见例子：

- 给机械臂相机做标定
- 去除广角镜头畸变
- 做双目测距前先求相机参数

## 双目 / 3D 重建中的基础流程

OpenCV 在双目视觉里常负责匹配、校正、三角化等基础步骤。

```python
import cv2 # 导入 OpenCV

left_gray = cv2.imread("left.jpg", 0) # 读取左相机灰度图
right_gray = cv2.imread("right.jpg", 0) # 读取右相机灰度图
stereo = cv2.StereoBM_create(numDisparities=16 * 6, blockSize=15) # 创建块匹配视差计算器
disparity = stereo.compute(left_gray, right_gray) # 根据左右图计算每个像素的视差
```

常见例子：

- 根据左右图计算视差图
- 由视差估计深度
- 在三维重建流程里做立体校正和特征匹配

## 轮廓提取

如果你想拿到某个物体的边界，通常会先二值化，再找轮廓。

```python
import cv2

img = cv2.imread("shape.png", 0) # 读取灰度图
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # 先把图像变成黑白二值图
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 查找最外层轮廓
```

## 画框、画点、写文字

调试视觉算法时，经常需要把检测结果直接画在图上。

```python
import cv2

img = cv2.imread("cat.jpg") # 读取原图
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2) # 画一个绿色矩形框，线宽为 2
cv2.circle(img, (120, 120), 5, (0, 0, 255), -1) # 在指定位置画一个红色实心圆点
cv2.putText(img, "cat", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # 在图上写文字 cat
```


## 见的 OpenCV 流程

很多任务并不是只调用一个函数，而是一串操作组合起来：

```python
import cv2

img = cv2.imread("input.jpg") # 读取输入图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 先转成灰度图，减少后续处理复杂度
blur = cv2.GaussianBlur(gray, (5, 5), 0) # 对灰度图做高斯平滑，降低噪声
edges = cv2.Canny(blur, 50, 150) # 在平滑后的图上做边缘检测
cv2.imwrite("edges.jpg", edges) # 把边缘检测结果保存下来
```
