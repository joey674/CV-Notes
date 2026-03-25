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

{% code overflow="wrap" %}
```python
#!/usr/bin/env python3
from __future__ import annotations

#####################################################################################
# arg
#####################################################################################
# 文件夹名
SCENE_NAME = "rabbit"

#####################################################################################
# cuda/da3配置
#####################################################################################
import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

INPUT_DIR = ROOT / "input" / SCENE_NAME
OUTPUT_DIR = ROOT / "output" / SCENE_NAME

MODEL_DIR = "depth-anything/DA3-GIANT-1.1"
PROCESS_RES = 336
PROCESS_RES_METHOD = "lower_bound_resize"
REF_VIEW_STRATEGY = "saddle_balanced"
TRAJECTORY_MODE = "extend"
VIDEO_QUALITY = "medium"
VIS_DEPTH = "hcat"
CHUNK_SIZE = 1
GS_VIEWS_INTERVAL = 1

def _preload_conda_cxx_runtime() -> None:
    # 先加载 conda 自带的 C++ 运行时，避免 CUDA 扩展混用
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    for lib_name in ("libstdc++.so.6", "libgcc_s.so.1"):
        lib_path = os.path.join(conda_prefix, "lib", lib_name)
        if not os.path.exists(lib_path):
            continue
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
_preload_conda_cxx_runtime()

#####################################################################################
# 推理部分
#####################################################################################

import torch
import moviepy.editor as mpy

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.layout_helpers import hcat, vcat
from depth_anything_3.utils.visualize import vis_depth_map_tensor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_QUALITY_MAP = {
    "low": {"crf": "28", "preset": "veryfast"},
    "medium": {"crf": "23", "preset": "medium"},
    "high": {"crf": "18", "preset": "slow"},
}

def collect_images(input_dir: Path) -> list[str]:
    # 读取目录图片
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")

    image_paths = sorted(
        str(path)
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found under: {input_dir}")
    return image_paths


def save_ply(prediction, save_path: Path) -> None:
    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=str(save_path),
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=GS_VIEWS_INTERVAL,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )


def save_video(prediction, save_path: Path) -> None:
    gs_world = prediction.gaussians
    tgt_extrs = torch.from_numpy(prediction.extrinsics).unsqueeze(0).to(gs_world.means)
    if prediction.is_metric:
        scale_factor = prediction.scale_factor
        if scale_factor is not None:
            tgt_extrs[:, :, :3, 3] /= scale_factor
    tgt_intrs = torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(gs_world.means)
    height, width = prediction.depth.shape[-2:]

    color, depth = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(height, width),
        chunk_size=CHUNK_SIZE,
        trj_mode=TRAJECTORY_MODE,
        use_sh=True,
        color_mode="RGB+ED",
        enable_tqdm=True,
    )

    video = color[0]
    if VIS_DEPTH is not None:
        depth_vis = vis_depth_map_tensor(depth[0])
        cat_fn = hcat if VIS_DEPTH == "hcat" else vcat
        video = torch.stack([cat_fn(rgb, dep) for rgb, dep in zip(video, depth_vis)])

    frames = (video.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    ffmpeg_params = [
        "-crf",
        VIDEO_QUALITY_MAP[VIDEO_QUALITY]["crf"],
        "-preset",
        VIDEO_QUALITY_MAP[VIDEO_QUALITY]["preset"],
        "-pix_fmt",
        "yuv420p",
    ]
    clip = mpy.ImageSequenceClip(list(frames), fps=24)
    clip.write_videofile(
        str(save_path),
        codec="libx264",
        audio=False,
        fps=24,
        ffmpeg_params=ffmpeg_params,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")

    # 某些 torch/CUDA 组合下会被 cuDNN 触发段错误，因此关闭
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda")
    # 让扩展只针对当前 GPU 架构编译，不去覆盖一堆无关架构
    major, minor = torch.cuda.get_device_capability(device)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")

    # 创建同名输出目录
    image_paths = collect_images(INPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scene: {SCENE_NAME}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model: {MODEL_DIR}")
    print(f"Device: {device}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")
    print(f"Found {len(image_paths)} image(s)")
    if len(image_paths) < 2:
        print("Only one image was found; the renderer will use a wander trajectory.")

    # 加载 DA3 模型
    model = DepthAnything3.from_pretrained(MODEL_DIR).to(device=device)
    prediction = model.inference(
        image_paths,
        infer_gs=True,
        process_res=PROCESS_RES,
        process_res_method=PROCESS_RES_METHOD,
        ref_view_strategy=REF_VIEW_STRATEGY,
        export_dir=None,
    )

    if prediction.gaussians is None:
        raise RuntimeError("Inference completed, but no 3DGS output was produced.")

    video_file = OUTPUT_DIR / f"{SCENE_NAME}.mp4"
    ply_file = OUTPUT_DIR / f"{SCENE_NAME}.ply"

    # 直接把视频和 ply 保存到场景输出根目录。
    save_video(prediction, video_file)
    save_ply(prediction, ply_file)

    print("")
    print("3DGS demo finished.")
    print(f"Depth shape: {prediction.depth.shape}")
    print(f"Extrinsics shape: {prediction.extrinsics.shape}")
    print(f"Intrinsics shape: {prediction.intrinsics.shape}")
    print(f"Gaussian count: {prediction.gaussians.means.shape[1]:,}")
    print(f"Video: {video_file}")
    print(f"PLY: {ply_file}")


if __name__ == "__main__":
    main()

```
{% endcode %}

