# Open3D 常做什么

Open3D 是一个 3D 数据处理库，常用来做点云、网格、RGB-D、配准、重建和可视化

```python
import open3d as o3d
import numpy as np
```

```bash
pip install open3d
```

## 读取、保存点云

最基础的用途就是把点云读进来，再处理后保存。

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("cloud.ply") # 从文件读取点云
print(pcd) # 打印点云的基本信息
o3d.io.write_point_cloud("cloud_copy.ply", pcd) # 把点云保存到新文件
```

## 3D 可视化

Open3D 很常用来直接看点云、网格和坐标系。

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("cloud.ply") # 读取点云
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5) # 创建一个坐标轴
o3d.visualization.draw_geometries([pcd, coord]) # 打开窗口显示点云和坐标轴
```

## 点云下采样

点云太密时，通常会先下采样，降低计算量。

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("cloud.ply") # 读取原始点云
down = pcd.voxel_down_sample(voxel_size=0.05) # 用体素网格做下采样，0.05 是体素边长
o3d.io.write_point_cloud("cloud_down.ply", down) # 保存下采样后的点云
```

## 点云平移、旋转、缩放

很多 3D 任务都要对点云做坐标变换。

```python
import copy
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("cloud.ply") # 读取点云
pcd2 = copy.deepcopy(pcd) # 复制一份点云，避免改掉原始数据

pcd2.translate((1, 0, 0)) # 把点云沿 x 方向平移 1 个单位

R = pcd2.get_rotation_matrix_from_xyz((0, 0, np.pi / 4)) # 生成绕 z 轴旋转 45 度的旋转矩阵
pcd2.rotate(R, center=(0, 0, 0)) # 按给定旋转矩阵旋转点云

pcd2.scale(0.5, center=pcd2.get_center()) # 以点云中心为基准缩小到 0.5 倍
```

## 法向量估计

点云的法向量理解成：某个点附近那一小块“局部表面”朝向

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("cloud.ply") # 读取点云
search = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30) # 设置法向量估计时的邻域搜索参数
pcd.estimate_normals(search_param=search) # 为每个点估计法向量
pcd.normalize_normals() # 把法向量归一化成单位向量
```

常见例子：

- 点到平面 ICP
- 泊松重建前先估计法向量
- 计算表面朝向

## 去噪、去离群点

点云里常会有噪声点，可以先做过滤。

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("cloud.ply") # 读取点云
filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0) # 按统计方法去掉离群点
outlier_pcd = pcd.select_by_index(ind, invert=True) # 取出被判定为离群点的部分
```

## ICP 配准

Open3D 很常被拿来做点云对齐，最经典的是 ICP

```python
import numpy as np
import open3d as o3d

source = o3d.io.read_point_cloud("source.ply") # 读取源点云
target = o3d.io.read_point_cloud("target.ply") # 读取目标点云

trans_init = np.eye(4) # 创建一个 4x4 单位矩阵作为初始位姿
distance_threshold = 0.02 # 设置匹配时允许的最大对应点距离
estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint() # 使用点到点误差模型
result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, trans_init, estimation) # 执行 ICP 配准
source.transform(result.transformation) # 用求出的变换矩阵把源点云对齐到目标点云
```


## 网格读取、处理

除了点云，Open3D 也能直接处理三角网格。

```python
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("mesh.ply") # 读取三角网格
mesh.compute_vertex_normals() # 计算顶点法向量，渲染时会更自然
mesh.paint_uniform_color([0.7, 0.7, 0.7]) # 给整个网格设置统一颜色
o3d.io.write_triangle_mesh("mesh_copy.ply", mesh) # 保存处理后的网格
```

常见例子：

- 查看重建得到的 mesh
- 给 mesh 上色后可视化
- 导出处理后的网格结果

## RGB-D 转点云

如果有彩色图和深度图，可以直接生成点云。

```python
import open3d as o3d

color = o3d.io.read_image("color.png") # 读取彩色图
depth = o3d.io.read_image("depth.png") # 读取深度图
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth) # 把彩色图和深度图打包成 RGB-D 数据

intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault) # 创建一个默认针孔相机内参对象
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic) # 根据 RGB-D 和内参生成点云
```

## 常见的 Open3D 流程

很多任务是一串操作连起来

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("cloud.ply") # 读取原始点云
pcd = pcd.voxel_down_sample(voxel_size=0.05) # 先做体素下采样
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0) # 再做离群点过滤
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)) # 估计法向量
o3d.visualization.draw_geometries([pcd]) # 显示处理后的点云
```