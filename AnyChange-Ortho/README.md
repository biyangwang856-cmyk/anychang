# AnyChange-Ortho

这是两期 GeoTIFF 的全幅变化检测工具。输入可以是无人机正射图，也可以是同一区域其他带坐标的 8 位 RGB/RGBA GeoTIFF。程序读取地理坐标，把两幅图转换到同一个米制网格，处理两期共同有效的区域。仅有 GPS 拍摄点的透视照片需要先完成地理配准或正射校正。

### 中文快速开始

1. 解压后进入 `AnyChange-Ortho`。使用 64 位 Python 3.10 或 3.11，按下方说明安装 PyTorch 和 `requirements.txt` 中的依赖。默认 ViT-H 建议预留至少 8 GB 空闲显存。
2. 把较早图像放在第一个参数，较晚图像放在第二个参数。输入路径有空格时加引号。结果目录须为空或尚未创建。

```bash
python run.py "T1.tif" "T2.tif" --out results
```

3. 打开结果中的 `viewer.html` 看离线卷帘，用 `report.html` 查看或打印报告。GIS 软件可读取 `change_mask.tif` 和 `change_candidates.geojson`，CSV 含各候选的面积和代表点。

首次运行默认下载并校验 SAM ViT-H 权重（约 2.56 GB）。实验室或离线环境已有权重时，添加 `--weights /path/to/sam_vit_h_4b8939.pth`。默认分辨率取 0.15 米和输入中较粗分辨率的较大值。沿用此前工程的像元大小时，可显式设定 `--resolution 0.15`。已有可靠地理配准时，可用 `--alignment geo` 直接按地理坐标对齐。

模型输出可包含车辆、阴影和视差造成的候选。ZIP 面向源码分享，包含上游源码、许可证和环境说明，影像、权重和结果由使用者各自准备。

AnyChange-Ortho detects candidate changes between two georeferenced RGB GeoTIFFs of the same site. The runner aligns them on a common grid, runs AnyChange on their shared valid coverage, and exports GIS results, an offline comparison viewer, and a printable HTML report.

## Install

Use 64-bit Python 3.10 or 3.11. For a GPU run, use an NVIDIA GPU with a driver compatible with the installed PyTorch CUDA build. Start with at least 8 GB of available GPU memory for the default SAM ViT-H settings. PyTorch 2.4.0 and torchvision 0.19.0 with CUDA 12.1 are the versions used for the AnyChange run on a Linux NVIDIA H20.

Open a terminal in the extracted `AnyChange-Ortho` directory and create a virtual environment.

**Linux**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell restricts script activation, use `.\.venv\Scripts\python.exe` wherever the following commands use `python`. For Python 3.11, change `3.10` to `3.11` in the environment command.

Install the PyTorch CUDA wheels before the remaining packages. See the [PyTorch install commands](https://pytorch.org/get-started/previous-versions/#v240) for other builds.

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last command should print `True` for a CUDA setup. The runner uses bundled AnyChange and SAM source files. For a CPU-only environment, use the same setup with `https://download.pytorch.org/whl/cpu` as the wheel index. CPU inference is most practical for small image pairs.

## Run on your images

```bash
python run.py T1.tif T2.tif --out results
```

Use `T1` for the earlier image and `T2` for the later one. Use paths in quotes if they contain spaces. Inputs must be 8-bit RGB GeoTIFFs, with an optional fourth alpha band, valid georeferencing, and overlapping ground coverage. For other bit depths or band layouts, export an 8-bit RGB or RGBA copy and preserve the NoData mask or alpha. The runner works on the common valid footprint. Choose a new or empty output directory.

The first run downloads the official checkpoint for the selected backbone, verifies its SHA-256, and stores it under `~/.cache/anychange-ortho`. Use `--cache-dir` to choose another location. You can run offline with a local checkpoint by adding `--weights /path/to/model.pth`. Use `sam_vit_h_4b8939.pth` for `vit_h` or `sam_vit_b_01ec64.pth` for `vit_b`. The same checksum check applies to local files.

```bash
python run.py "images/early.tif" "images/late.tif" --out results --resolution 0.15 --alignment auto --device cuda:0
```

For a small CPU run, choose the lighter model and lower the prompt count.

```bash
python run.py T1.tif T2.tif --out results-cpu --device cpu --backbone vit_b --points-per-side 8 --points-per-batch 8
```

Changing the backbone or prompt count changes the candidate map. Use these settings for a quick inspection and compare the results before applying them to a larger job.

| Option | Use |
| --- | --- |
| `--resolution METRES` | Target metres per pixel. If omitted, use the coarser of 0.15 m and the input ground resolution. |
| `--alignment auto` or `geo` | `auto` checks matching image features and estimates an adjustment when the matches support it. `geo` uses the supplied georeferencing. |
| `--prepare-only` | Generate aligned inputs, the quality layer and `alignment.json` for inspection without running AnyChange. |
| `--device DEVICE` | `auto` uses CUDA when available and otherwise runs on CPU. Use `cuda:0` to require the first CUDA device, or `cpu` to select CPU explicitly. |
| `--backbone vit_h` or `vit_b` | Select the SAM encoder. `vit_h` is the default. |
| `--angle DEGREES` | AnyChange change threshold. The default is 145°. |
| `--tile-size PIXELS` and `--stride PIXELS` | Change the inference window and step. Defaults are 256 and 192 pixels. |
| `--points-per-side N` and `--points-per-batch N` | Control SAM grid prompts and the number processed together. Defaults are 32 and 16. Lower the batch size if GPU memory is tight. |
| `--min-area M2` | Minimum connected region area for the vector export. The default is 4 m². |
| `--weights FILE` and `--cache-dir DIRECTORY` | Use a local checkpoint or choose where downloaded checkpoints are cached. |

Run `python run.py --help` for the complete CLI. The prepared grid is limited to 40 million pixels. If the runner reports that the grid is too large, select a larger `--resolution` value or crop the inputs to the area you want to compare.

## Results

| Output | Content |
| --- | --- |
| `t1_aligned.tif`, `t2_aligned.tif`, `quality.tif`, `alignment.json` | T1 and T2 on the common grid, the registration quality layer, and alignment metadata. |
| `anychange_score.tif`, `change_mask.tif` | Per-pixel score and original thresholded candidates. In the mask, 0 means no candidate, 1 means candidate, and 255 means outside common coverage. |
| `change_candidates.geojson`, `change_candidates.csv` | Candidate polygons and area estimates after the minimum-area filter. |
| `viewer.html` | Offline comparison of T1, T2 and the candidate overlay. |
| `summary.json`, `report.html` | Run settings, coverage, candidate statistics and a printable report. Open `report.html` in a browser and print to PDF if needed. |

Candidate regions need visual review. A large candidate area can include differences from vehicles, shadows, occlusion, reflections, or stitching. Review the aligned images and registration evidence before interpreting changes near roofs, trees, roads, or coverage boundaries. Use the GeoTIFF or vector data for measurements. The browser overview may be downsampled for display.

## Share the source

The release is designed to be extracted and run with user-supplied GeoTIFFs. Commit the code, README and third-party license files to your GitHub repository. The `.gitignore` excludes common locations for orthophotos, model weights and generated results. Keep your own data and output folders local unless you intend to publish them.

## Credits

- **AnyChange** · Zhuo Zheng, Yanfei Zhong, Liangpei Zhang and Stefano Ermon. *Segment Any Change*, NeurIPS 2024. [Paper](https://arxiv.org/abs/2402.01188) · [Pinned upstream source](https://github.com/Z-Zheng/pytorch-change-models/tree/190b07a2eb4c389a5dc3f72802c67f428cdfd6b7)
- **SAM** · Alexander Kirillov and colleagues. *Segment Anything*, ICCV 2023. [Paper](https://arxiv.org/abs/2304.02643) · [Official source and checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

The bundled upstream AnyChange and SAM source is distributed under Apache License 2.0. See the [vendored license](vendor/LICENSE) and [upstream notice](vendor/NOTICE.md) for the included files and local adaptations.
