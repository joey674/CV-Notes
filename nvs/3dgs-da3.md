# 3DGS-DA3

## Overview

先聊聊结果;&#x20;

发现整个推理速度极快, 但是太吃显存;&#x20;

* 首先large/giant才支持GS,本身需要运行1.15-1.3B的参数模型;&#x20;
* 其次GS-head输出默认是开启gsplat渲染出结果 (如果不输出ply video的话 避免使用gsplat,那显存占用应该还好),也吃cuda显存

其次效果一般,&#x20;

* 会有大量散点在远距离生成;&#x20;
* 但是近处(置信度高的地方)生成效果还是很好的

部署稍微麻烦一点, 需要配置死一套 不然容易崩溃(conda torch gsplat不同的C++/CUDA拓展)

## Deployment and Code

部署

```bash
# 首先clone da3的仓库
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd 
```

```bash
conda create -n da3 python=3.10.19 -y
conda activate da3

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  "numpy<2" \
  "torch==2.10.0" \
  "torchvision==0.25.0" \
  "xformers==0.0.35"

conda install -y -c nvidia cuda-nvcc=12.8.93

export CUDA_HOME=$CONDA_PREFIX #
export PATH=$CUDA_HOME/bin:$PATH #
unset LD_LIBRARY_PATH

python -m pip install -e ".[app]"

# 只编译特定版本gsplat(对于4070ti)
export TORCH_CUDA_ARCH_LIST="8.9"
env -u LD_LIBRARY_PATH python -m pip install --no-build-isolation \
  "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"
```

```
python - <<'PY'
import torch, xformers
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("xformers:", xformers.__version__)
print("xformers cpp:", getattr(xformers, "_has_cpp_library", None))
PY
```



