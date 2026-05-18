#!/usr/bin/env python3
"""
演示脚本

实时摄像头或视频文件的跌倒风险评估演示

用法:
    python scripts/demo.py --source camera --camera-id 0
    python scripts/demo.py --source video --path input.mp4
    python scripts/demo.py --source ezviz --device-serial XXXXX
"""
import sys
import argparse
import cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CFG_TRAIN
from src.inference.pipeline import InferencePipeline
from src.utils.visualization import draw_keypoints, draw_risk_overlay
from src.ezviz.event_handler import EventHandler


def main():
    parser = argparse.ArgumentParser(description="跌倒风险演示")
    parser.add_argument("--source", default="camera", choices=["camera", "video", "ezviz"])
    parser.add_argument("--path", help="视频文件路径")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--device-serial", help="萤石设备序列号")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=CFG_TRAIN.DEVICE)
    parser.add_argument("--output", default=None, help="输出视频路径")
    args = parser.parse_args()

    # 初始化流水线
    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # 事件处理器
    event_handler = EventHandler()

    # 确定视频源
    if args.source == "camera":
        source = args.camera_id
    elif args.source == "video":
        source = args.path
    elif args.source == "ezviz":
        from src.ezviz.api_client import EZVIZClient
        from src.ezviz.stream import VideoStreamHandler
        client = EZVIZClient()
        stream = VideoStreamHandler(client, args.device_serial)
        stream.start()
        source = None  # 从 stream 获取帧
    else:
        print(f"未知来源: {args.source}")
        return

    # 输出视频
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 30, (640, 480))

    print("[Demo] 启动演示，按 'q' 退出")

    try:
        if args.source == "ezviz":
            # 萤石模式
            while True:
                frame = stream.get_frame()
                if frame is None:
                    continue
                result = pipeline.process_frame(frame)
                if result:
                    event_handler.handle(result)
                    frame = draw_risk_overlay(frame, result)
                cv2.imshow("Fall Risk Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # 视频/摄像头模式
            for item in pipeline.process_video(source):
                frame = item["frame"]
                result = item["result"]
                if result:
                    event_handler.handle(result)
                    frame = draw_risk_overlay(frame, result)
                cv2.imshow("Fall Risk Detection", frame)
                if writer:
                    writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()
        if writer:
            writer.release()
        if args.source == "ezviz":
            stream.stop()

        # 打印统计
        stats = event_handler.get_statistics()
        print(f"\n[Demo] 统计: {stats}")


if __name__ == "__main__":
    main()
