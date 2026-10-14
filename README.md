# AI Watermark Remover (PySide6)

本地运行的 AI 水印擦除工具，支持图片与视频，支持移动/渐隐渐现的动态水印。
图形界面基于 PySide6，融合两个优秀开源项目的思路：

- [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) —— 文本/字幕检测 (PaddleOCR) + 时序修复 (STTN/ProPainter/LaMa)
- [D-Ogi/WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI) —— Florence-2 自动检测 + LaMa 修复 + 隔帧检测/淡入淡出

> 免责声明：本工具仅用于去除您拥有合法权利或已获授权的内容中的水印。请勿用于侵犯他人版权。

## 功能

- 输入：图片 (PNG/JPG/WEBP/BMP/TIFF)、视频 (MP4/AVI/MOV/MKV/FLV/WMV/WEBM)
- 水印区域确定：手动框选/涂抹、Florence-2 自动检测、PaddleOCR 文本检测（可“预览”后回填画布微调）
- 动态水印：关键帧标注 + 光流/跟踪传播 + 隔帧重检测 + 淡入淡出扩展
- 修复引擎：LaMa（图片/逐帧）、STTN、ProPainter（视频时序）、OpenCV（快速兜底）
- NVIDIA CUDA 加速，自动降级 CPU
- 后台多线程处理，进度显示与取消；设置持久化；整文件夹批处理

## 目录结构

```
app.py                      # GUI 入口
requirements.txt
src/
  gui/       main_window, mask_canvas, video_timeline, settings_panel, workers, preview_worker
  core/      types, mask_propagation, pipeline_image, pipeline_video
  detection/ base, manual, florence2, paddleocr_det, factory
  inpaint/   base, lama, sttn, propainter, opencv_fallback, factory, errors, _vendor/
  models/    paths, downloader
  utils/     device, config, logger, video_io
tests/       冒烟测试
```

## 安装

建议使用独立虚拟环境。Python 3.10+（推荐 3.12）。

### 1) NVIDIA GPU (CUDA)

先按你的 CUDA 版本安装 PyTorch 与 Paddle（示例为 CUDA 12.1 / 11.8）：

```bash
python -m venv wmrEnv
# Windows PowerShell:
wmrEnv\Scripts\Activate.ps1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install -r requirements.txt
```

> CUDA 11.8 用户把 torch 的 `cu121` 换成 `cu118`，paddle 换成对应版本即可。

### 2) 仅 CPU / 无 NVIDIA 显卡

```bash
pip install torch torchvision torchaudio
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements.txt
```

Windows 上 AMD/Intel 独显/集显可另装 `torch_directml`（需自行适配 `device.py`）。

### 3) FFmpeg（可选，强烈推荐）

处理视频并保留原音轨需要 FFmpeg。将其加入 PATH，程序会自动通过 FFmpeg 管道编码并 mux 原音频；
无 FFmpeg 时回退到 OpenCV `VideoWriter`（不保留音频）。

## 运行

```bash
python app.py
```

首次运行建议先用菜单 **Models → Download LaMa** 下载 LaMa 权重（图片/逐帧修复的默认引擎）。
权重缓存于 `~/.watermarkremoval/models`（可用环境变量 `WMR_MODEL_DIR` 覆盖）。

## 使用流程

### 图片
1. File → Open file，选择图片
2. Detection Method 选 **Manual**，用 Rectangle / Brush 涂抹水印
3. Inpaint engine 选 **LaMa**，点击 **Start removal**
4. 输出默认在源文件旁，命名 `<原名>_nowm.<ext>`（可用 File → Set output folder 指定）

### 视频（含动态水印）
1. 打开视频，时间轴出现
2. Manual：在若干关键帧上涂抹水印（切换帧会自动保存该帧 mask）
   Auto：选择 Florence-2 / PaddleOCR，设 “Detect every N frames”，可先 **Preview detection**
3. Propagation：Auto / Optical flow / Tracking 用于让 mask 跟随移动水印；
   Fade-in / Fade-out（秒）用于覆盖渐隐渐现水印
4. 引擎：静态或轻度移动水印用 LaMa（逐帧）；明显时序闪烁用 STTN；剧烈运动用 ProPainter
5. Start removal → 后台逐帧/分段修复并合成（保留音轨）

## 时序引擎 (STTN / ProPainter) 说明

STTN 与 ProPainter 的网络结构与官方权重属于大型第三方模型，本仓库不内置其源码。
若要启用，请将上游实现放入 `_vendor` 目录，各提供一个入口：

```
src/inpaint/_vendor/sttn_impl.py       def load_model(weights_path, device) -> callable(frames, mask) -> list[frames]
src/inpaint/_vendor/propainter_impl.py def load_model(weights_path, device) -> callable(frames, masks) -> list[frames]
```

- 官方权重下载：菜单 Models → Download STTN / Download ProPainter
- 若 `_vendor` 中缺少对应模块，选中该引擎时会自动 **降级为 LaMa 逐帧**（日志会给出提示），任务仍可完成
- 参考实现来源：`microsoft/StTN`、`YaoFANGUK/video-subtitle-remover`、`sczhou/ProPainter`

## 自动检测依赖

- Florence-2：需要 `transformers`；首次从 HuggingFace 拉取 `microsoft/Florence-2-base`
- PaddleOCR：需要 `paddlepaddle` + `paddleocr`；仅适合文字型水印/字幕

任一不可用时会在日志中报错，不影响手动模式与其它流程。

## 常见问题

- **速度慢**：图片/逐帧可选 LaMa super-fast；视频增大 “Detect every N frames”、
  降低分辨率或使用 STTN 跳过逐帧检测。
- **视频逐帧效果闪烁**：改用 STTN/ProPainter，或增大 STTN neighbor/reference 参数。
- **无音频输出**：确认已安装并在 PATH 中配置 FFmpeg。
- **CUDA 不可用**：程序自动回退 CPU 并在右上角显示当前设备。

## 打包为 exe 与自动发布

默认打包为 onefile 单文件 exe（免安装）。本地手动打包：

```powershell
# CPU 版（单 exe）
pip install -r requirements-build-cpu.txt
pyinstaller --noconfirm watermark_removal.spec
# 产物：dist\WatermarkRemover.exe

# 如需 CUDA 版（建议 onedir，单文件会超过 GitHub Release 2GB 上限）
pip install -r requirements-build-cuda.txt
$env:WMR_BUILD_MODE="onedir"
pyinstaller --noconfirm watermark_removal.spec
# 产物：dist\WatermarkRemover\WatermarkRemover.exe
```

GitHub Actions 自动发布（推荐）：

- 触发方式：推送形如 `v0.1.0` 的 tag，或在 Actions 页面手动 `Build Windows Release`（`workflow_dispatch`）
- 构建矩阵：同时产出 `cpu`（onefile 单 exe）与 `cuda`（onedir 文件夹压缩包）两个变体，
  分别用 [requirements-build-cpu.txt](requirements-build-cpu.txt) / [requirements-build-cuda.txt](requirements-build-cuda.txt) 安装依赖，并用 [watermark_removal.spec](watermark_removal.spec) 打包
- 发布：构建任务上传各自 artifact，随后 `release` 任务汇总为一次 GitHub Release，
  附带 `WatermarkRemover-win64-cpu.zip` 与 `WatermarkRemover-win64-cuda.zip`
  （tag 触发为正式版；手动触发默认 pre-release）
- 需要仓库已授予工作流 `contents: write` 权限（本文件已声明）

> 想改回单文件 GPU 版：把矩阵里 `cuda` 的 `build_mode` 改为 `onefile` 即可（注意可能超限）。

图标与体积：

- **应用图标**：无需提交二进制。构建时 [watermark_removal.spec](watermark_removal.spec) 会调用
  [assets/make_icon.py](assets/make_icon.py)（基于 Pillow）生成 `assets/app.ico`，同时作为 exe 文件图标
  与运行时窗口/任务栏图标。可手动运行 `python assets/make_icon.py` 生成 `app.png` 预览。想换自定义图标，
  只需把 `assets/app.ico` 换成你自己的（存在则直接使用，不再生成）。
- **UPX 压缩**：默认开启以显著减小体积（CI 已用 Chocolatey 安装 upx）。设环境变量 `WMR_UPX=0` 可关闭；
  Qt/运行库等关键 DLL 已加入排除名单以避免兼容/安全软件误报。

打包取舍：便携版内置核心链路（PySide6 + OpenCV + NumPy + CPU 版 torch/LaMa）。
为保证体积与打包稳定，Florence-2（transformers）、PaddleOCR（paddle）、
STTN/ProPainter 未打包，选用时会在日志中提示并自动降级；LaMa 权重首次运行仍联网下载。
如需 GPU/全部检测能力，请按上文在本地用完整 `requirements.txt` 源码运行。

## 开发与测试

```bash
python -m compileall -q src app.py     # 语法编译检查
pytest -q                              # 运行 tests/ 下的冒烟测试（需已装 numpy/opencv/PySide6）
```
