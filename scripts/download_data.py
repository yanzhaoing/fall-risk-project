#!/usr/bin/env python3
"""
数据集准备脚本

说明:
- 该脚本会在仓库内创建标准数据目录
- 输出每个数据集的官方或当前可用下载入口
- 不会把大型原始数据直接提交进 GitHub 仓库

用法:
    python scripts/download_data.py --dataset upfall
    python scripts/download_data.py --dataset ntu
    python scripts/download_data.py --dataset all
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CFG_PATHS, CFG_DATA

DATASETS = {
    "upfall": {
        "name": "UP-Fall Detection Dataset",
        "target_dir": CFG_DATA.UPFALL_DIR,
        "description": "多模态跌倒检测数据集（RGB + Depth + IR + IMU）",
        "download_url": "https://www.mdpi.com/1424-8220/19/9/1988",
        "access": "论文页与数据说明页，按提供方式申请或下载",
        "notes": [
            "当前仓库已有 UPFallDataset，建议原始数据放在该目录下。",
            "如果后续做关键点预处理，可把中间产物放到 data/processed。",
        ],
    },
    "le2i": {
        "name": "Le2i Fall Detection Dataset",
        "target_dir": CFG_DATA.LE2I_DIR,
        "description": "室内场景视频跌倒检测数据集（RGB）",
        "download_url": "https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia",
        "access": "当前更稳定的公开入口是 Kaggle 镜像",
        "notes": [
            "适合做 RGB 视频跌倒检测基线和场景对比实验。",
            "建议保留原始视频在该目录下，处理后的帧或关键点放到 data/processed。",
        ],
    },
    "ntu": {
        "name": "NTU RGB+D / NTU RGB+D 120",
        "target_dir": CFG_DATA.NTU_DIR,
        "description": "大规模 RGB+D / skeleton 动作识别数据集",
        "download_url": "https://rose1.ntu.edu.sg/dataset/actionRecognition/",
        "access": "官方页面申请后下载",
        "notes": [
            "当前训练脚本默认读取 data/processed/ntu_coco。",
            "原始 NTU 数据建议先放在该目录，再做骨架/关键点转换。",
            "转换后的 keypoints.json 建议放在 data/processed/ntu_coco。",
        ],
    },
    "urfall": {
        "name": "UR Fall Detection Dataset",
        "target_dir": CFG_DATA.URFALL_DIR,
        "description": "深度图 + 加速度的跌倒检测数据集",
        "download_url": "https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
        "access": "官方公开下载页面",
        "notes": [
            "适合补充多模态小规模验证。",
            "建议保留官方原始目录结构，避免后续对齐脚本变复杂。",
        ],
    },
    "etri": {
        "name": "ETRI-Activity3D",
        "target_dir": CFG_DATA.ETRI_DIR,
        "description": "面向老年人日常行为识别的大规模 RGB-D 数据集",
        "download_url": "https://ai4robot.github.io/etri-activity3d-en/",
        "access": "官方数据集页面",
        "notes": [
            "更适合做老年人日常行为先验、正常行为建模和域迁移。",
            "如果用于当前项目，建议先抽取与跌倒风险高度相关的动作子集。",
        ],
    },
}


def ensure_structure():
    """创建标准目录结构"""
    dirs = [
        CFG_PATHS.DATA_DIR,
        CFG_PATHS.DATASETS_DIR,
        CFG_PATHS.RAW_DIR,
        CFG_PATHS.PROCESSED_DIR,
        CFG_DATA.UPFALL_DIR,
        CFG_DATA.LE2I_DIR,
        CFG_DATA.NTU_DIR,
        CFG_DATA.URFALL_DIR,
        CFG_DATA.ETRI_DIR,
        CFG_DATA.NTU_PROCESSED_DIR,
    ]
    for path in dirs:
        Path(path).mkdir(parents=True, exist_ok=True)


def print_dataset_info(dataset_name: str):
    if dataset_name not in DATASETS:
        print(f"未知数据集: {dataset_name}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return

    info = DATASETS[dataset_name]
    target_dir = Path(info["target_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"数据集: {info['name']}")
    print(f"目标目录: {target_dir}")
    print(f"说明: {info['description']}")
    print(f"下载入口: {info['download_url']}")
    print(f"访问方式: {info['access']}")
    print("放置建议:")
    for note in info["notes"]:
        print(f"- {note}")
    print(f"{'=' * 72}")


def main():
    parser = argparse.ArgumentParser(description="准备仓库内数据集目录并输出下载入口")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=list(DATASETS.keys()) + ["all"],
        help="选择要准备的数据集",
    )
    args = parser.parse_args()

    ensure_structure()

    if args.dataset == "all":
        for name in DATASETS:
            print_dataset_info(name)
        print(f"\nNTU 预处理目录: {CFG_DATA.NTU_PROCESSED_DIR}")
    else:
        print_dataset_info(args.dataset)


if __name__ == "__main__":
    main()
