#!/usr/bin/env python3
"""
数据集下载脚本

用法:
    python scripts/download_data.py --dataset upfall
    python scripts/download_data.py --dataset le2i
    python scripts/download_data.py --all
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CFG_DATA

# 数据集下载信息
DATASETS = {
    "upfall": {
        "name": "UP-Fall Detection Dataset",
        "url": "https://researchdata.ntu.edu.sg/api/access/datafile/1376erta",
        "description": "多模态跌倒检测数据集（RGB+Depth+IR+IMU）",
        "size": "~15GB",
        "paper": "UP-Fall: A Dataset for Fall Detection (Sensors, 2019)",
    },
    "le2i": {
        "name": "Le2i Fall Detection Dataset",
        "url": "https://project.liris.fr/datasets/le2i-fall",
        "description": "室内场景跌倒检测数据集（RGB）",
        "size": "~2GB",
        "paper": "Evaluation of a Fall Detection System (AVSS, 2014)",
    },
}


def download_dataset(dataset_name: str):
    """下载指定数据集"""
    if dataset_name not in DATASETS:
        print(f"未知数据集: {dataset_name}")
        print(f"可用数据集: {list(DATASETS.keys())}")
        return

    info = DATASETS[dataset_name]
    print(f"\n{'=' * 50}")
    print(f"  数据集: {info['name']}")
    print(f"  描述: {info['description']}")
    print(f"  大小: {info['size']}")
    print(f"  论文: {info['paper']}")
    print(f"  下载地址: {info['url']}")
    print(f"{'=' * 50}")

    print(f"\n[注意] 请手动下载数据集并放置到:")
    print(f"  {CFG_DATA.DATASETS_DIR / dataset_name}")

    # 自动下载（如果 URL 可用）
    # TODO: 实现自动下载逻辑
    print("\n[TODO] 自动下载功能待实现")
    print("请根据上述链接手动下载数据集")


def main():
    parser = argparse.ArgumentParser(description="数据集下载")
    parser.add_argument(
        "--dataset",
        default="upfall",
        choices=list(DATASETS.keys()) + ["all"],
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            download_dataset(name)
    else:
        download_dataset(args.dataset)


if __name__ == "__main__":
    main()
