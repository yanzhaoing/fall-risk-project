#!/usr/bin/env python3
"""
数据集准备 / 下载 / 解压脚本

支持三类流程：
1. 自动下载：当前可直接支持 Le2i（Kaggle CLI）与 UR Fall（官方页面抓取）
2. 半自动下载：NTU、ETRI、UP-Fall 会准备目录并输出申请 / 手动下载说明
3. 本地解压：对手动下载后的 zip / tar.* 归档做批量解压

常用用法：
    python scripts/download_data.py --dataset all --prepare
    python scripts/download_data.py --dataset le2i --download
    python scripts/download_data.py --dataset urfall --download
    python scripts/download_data.py --dataset ntu --instructions
    python scripts/download_data.py --dataset ntu --extract
"""
import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CFG_DATA, CFG_PATHS


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href")
        if href:
            self.links.append(href)


DATASETS = {
    "upfall": {
        "name": "UP-Fall Detection Dataset",
        "mode": "manual",
        "target_dir": CFG_DATA.UPFALL_DIR,
        "download_url": "https://sites.google.com/up.edu.mx/challenge-up-2019/data",
        "reference_url": "https://www.mdpi.com/1424-8220/19/9/1988",
        "description": "多模态跌倒检测数据集（RGB + Depth + IR + IMU）",
        "instructions": [
            "当前公开挑战页可查看数据说明，但原始大文件入口并不稳定。",
            "建议先从论文与 Challenge UP 页面确认可用下载入口或联系作者。",
            "把你拿到的原始压缩包放进目标目录后，可重新运行本脚本的 --extract。",
        ],
    },
    "le2i": {
        "name": "Le2i Fall Detection Dataset",
        "mode": "kaggle",
        "target_dir": CFG_DATA.LE2I_DIR,
        "download_url": "https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia",
        "description": "室内场景视频跌倒检测数据集（RGB）",
        "kaggle_dataset": "tuyenldvn/falldataset-imvia",
        "instructions": [
            "需要本机已安装 Kaggle CLI，并配置好 Kaggle 凭证。",
            "如果 Kaggle 无法访问，也可以手动下载压缩包后放进目标目录，再运行 --extract。",
        ],
    },
    "ntu": {
        "name": "NTU RGB+D / NTU RGB+D 120",
        "mode": "request",
        "target_dir": CFG_DATA.NTU_DIR,
        "download_url": "https://rose1.ntu.edu.sg/dataset/actionRecognition/",
        "description": "大规模 RGB+D / skeleton 动作识别数据集",
        "instructions": [
            "需要先在官方页面按要求申请下载权限。",
            "原始压缩包建议先放入 data/datasets/NTU-RGBD/。",
            "处理后的 keypoints.json 建议放入 data/processed/ntu_coco/。",
            "拿到压缩包后可以运行本脚本的 --extract 来统一解压。",
        ],
    },
    "urfall": {
        "name": "UR Fall Detection Dataset",
        "mode": "official_scrape",
        "target_dir": CFG_DATA.URFALL_DIR,
        "download_url": "https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
        "description": "深度图 + 加速度的跌倒检测数据集",
        "instructions": [
            "脚本会从官方页面抓取 zip / csv 链接并下载到目标目录。",
            "下载完成后，如需展开压缩包，可继续执行 --extract。",
        ],
    },
    "etri": {
        "name": "ETRI-Activity3D",
        "mode": "membership",
        "target_dir": CFG_DATA.ETRI_DIR,
        "download_url": "https://ai4robot.github.io/etri-activity3d-en/",
        "membership_url": "https://nanum.etri.re.kr/share/list?lang=En_us",
        "eula_url": "https://ai4robot.github.io/resources/EULA_ETRIActivity3D_en.pdf",
        "description": "面向老年人日常行为识别的大规模 RGB-D 数据集",
        "instructions": [
            "完整数据集需要先加入官方分享页面并接受相应条款。",
            "官方页面提供样例下载，完整数据需要成员访问权限。",
            "建议先把手动下载得到的归档文件放入目标目录，再运行 --extract。",
        ],
    },
}

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def ensure_structure():
    paths = [
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
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def print_header(text: str):
    print(f"\n{'=' * 72}")
    print(text)
    print(f"{'=' * 72}")


def print_dataset_info(name: str):
    info = DATASETS[name]
    target_dir = Path(info["target_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)
    print_header(f"数据集: {info['name']}")
    print(f"目标目录: {target_dir}")
    print(f"下载方式: {info['mode']}")
    print(f"说明: {info['description']}")
    print(f"下载入口: {info['download_url']}")
    if info.get("membership_url"):
        print(f"成员入口: {info['membership_url']}")
    if info.get("eula_url"):
        print(f"EULA: {info['eula_url']}")
    print("操作提示:")
    for item in info.get("instructions", []):
        print(f"- {item}")


def require_kaggle_cli():
    if shutil.which("kaggle") is None:
        raise RuntimeError("未找到 kaggle 命令，请先安装 Kaggle CLI 并配置凭证。")


def run_kaggle_download(dataset_slug: str, target_dir: Path):
    require_kaggle_cli()
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", str(target_dir),
        "--unzip",
    ]
    print("执行:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
        size_mb = downloaded / (1024 * 1024)
        print(f"已下载: {output_path.name} ({size_mb:.1f} MB)")


def scrape_links(url: str):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    parser = LinkCollector()
    parser.feed(resp.text)
    return [urljoin(url, href) for href in parser.links]


def download_urfall(target_dir: Path):
    base_url = DATASETS["urfall"]["download_url"]
    all_links = scrape_links(base_url)
    file_links = []
    for link in all_links:
        lower = link.lower()
        if lower.endswith(".zip") or lower.endswith(".csv"):
            file_links.append(link)

    if not file_links:
        raise RuntimeError("没有从 UR Fall 官方页面解析到可下载文件。")

    print_header(f"开始下载 UR Fall，共 {len(file_links)} 个文件")
    for link in sorted(set(file_links)):
        filename = Path(link).name
        output = target_dir / filename
        if output.exists() and output.stat().st_size > 0:
            print(f"跳过已存在文件: {filename}")
            continue
        download_file(link, output)


def extract_archives(target_dir: Path):
    archives = []
    for path in sorted(target_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES):
            archives.append(path)

    if not archives:
        print(f"未在 {target_dir} 找到可解压归档文件。")
        return

    print_header(f"开始解压 {target_dir} 下的 {len(archives)} 个归档")
    for archive in archives:
        extract_dir = target_dir / archive.stem.replace(".tar", "")
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"解压: {archive.name} -> {extract_dir}")
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(extract_dir)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(extract_dir)
        else:
            print(f"跳过不支持的归档格式: {archive.name}")


def perform_download(dataset_name: str):
    info = DATASETS[dataset_name]
    target_dir = Path(info["target_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    if info["mode"] == "kaggle":
        run_kaggle_download(info["kaggle_dataset"], target_dir)
        return

    if info["mode"] == "official_scrape":
        download_urfall(target_dir)
        return

    print_dataset_info(dataset_name)
    print("\n该数据集当前不支持无交互自动下载。")
    print("你可以先手动获取压缩包，再运行：")
    print(f"python scripts/download_data.py --dataset {dataset_name} --extract")


def main():
    parser = argparse.ArgumentParser(description="准备、下载或解压项目数据集")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=list(DATASETS.keys()) + ["all"],
        help="选择要处理的数据集",
    )
    parser.add_argument("--prepare", action="store_true", help="仅准备目录并输出说明")
    parser.add_argument("--download", action="store_true", help="尝试自动下载支持的数据集")
    parser.add_argument("--extract", action="store_true", help="解压目标目录中的本地归档文件")
    parser.add_argument("--instructions", action="store_true", help="输出说明，不下载")
    args = parser.parse_args()

    ensure_structure()

    selected = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    did_action = False

    if args.prepare or args.instructions or not (args.download or args.extract):
        for name in selected:
            print_dataset_info(name)
        print(f"\nNTU 预处理目录: {CFG_DATA.NTU_PROCESSED_DIR}")
        did_action = True

    if args.download:
        for name in selected:
            perform_download(name)
        did_action = True

    if args.extract:
        for name in selected:
            extract_archives(Path(DATASETS[name]["target_dir"]))
        did_action = True

    if not did_action:
        parser.print_help()


if __name__ == "__main__":
    main()
