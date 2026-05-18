# 数据目录说明

本仓库已经为以下 5 个目标数据集预留了标准位置：

- `data/datasets/UP-Fall/`
- `data/datasets/Le2i/`
- `data/datasets/NTU-RGBD/`
- `data/datasets/UR-Fall/`
- `data/datasets/ETRI-Activity3D/`

另外，当前训练脚本默认使用的 NTU 关键点处理结果目录是：

- `data/processed/ntu_coco/`

## 放置原则

### 原始数据

请把官方原始下载内容放到对应的 `data/datasets/<DatasetName>/` 目录下，尽量保留原始目录结构。

### 中间处理结果

如果后续做关键点抽取、帧切分、标注转换或特征缓存，建议放到：

- `data/raw/`
- `data/processed/`

### GitHub 仓库注意事项

这些数据集体量较大，不应该把原始文件直接提交进 GitHub 仓库历史。当前 `.gitignore` 已经按这个原则配置好：

- 保留目录结构
- 忽略大型数据文件
- 允许占位目录被跟踪

## 当前目标数据集和目录对应

- UP-Fall → `data/datasets/UP-Fall/`
- Le2i → `data/datasets/Le2i/`
- NTU RGB+D → `data/datasets/NTU-RGBD/`
- UR Fall → `data/datasets/UR-Fall/`
- ETRI-Activity3D → `data/datasets/ETRI-Activity3D/`

## 当前脚本支持的流程

脚本位置：`scripts/download_data.py`

### 1. 仅准备目录并查看说明

```bash
python scripts/download_data.py --dataset all --prepare
```

### 2. 自动下载

当前可自动下载的数据集：

- `le2i`
  通过 Kaggle CLI 下载并自动解压。
- `urfall`
  从官方页面抓取 zip / csv 文件并下载。

示例：

```bash
python scripts/download_data.py --dataset le2i --download
python scripts/download_data.py --dataset urfall --download
```

### 3. 半自动流程

当前需要手动申请、登录或加入成员后再下载的数据集：

- `upfall`
- `ntu`
- `etri`

推荐流程：

1. 先运行说明命令，确认目标目录和入口：

```bash
python scripts/download_data.py --dataset ntu --instructions
```

2. 手动把拿到的压缩包放到对应目录。

3. 再运行自动解压：

```bash
python scripts/download_data.py --dataset ntu --extract
```

## 特别说明

### NTU RGB+D

- 原始数据建议放到 `data/datasets/NTU-RGBD/`
- 当前训练脚本默认读取 `data/processed/ntu_coco/`
- 所以后续还需要单独做一层骨架 / 关键点转换

### ETRI-Activity3D

- 完整数据集需要先加入官方分享页面并接受相应条款
- 官方页面还提供样例下载，但完整数据不是匿名直链

### UP-Fall

- 公开页面提供了论文与挑战说明，但原始大文件入口并不稳定
- 更适合作为“手动获取压缩包后，用本脚本统一解压和归位”的半自动流程
