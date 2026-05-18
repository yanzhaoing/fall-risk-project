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

## 配套脚本

可以使用下面的脚本查看下载入口并自动准备目录：

```bash
python scripts/download_data.py --dataset all
```

如果只想准备某一个数据集目录：

```bash
python scripts/download_data.py --dataset ntu
```
