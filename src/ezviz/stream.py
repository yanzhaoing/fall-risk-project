"""
视频流获取模块

从萤石摄像头获取实时视频流
支持 RTSP/RTMP 协议
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable

from .api_client import EZVIZClient


class VideoStreamHandler:
    """
    萤石摄像头视频流处理器

    在后台线程中持续读取视频帧，主程序可以随时取最新帧

    Args:
        client: EZVIZClient 实例
        device_serial: 设备序列号
        channel_no: 通道号
        buffer_size: 帧缓冲区大小
    """

    def __init__(
        self,
        client: EZVIZClient,
        device_serial: str,
        channel_no: int = 1,
        buffer_size: int = 5,
    ):
        self.client = client
        self.device_serial = device_serial
        self.channel_no = channel_no
        self.buffer_size = buffer_size

        self._cap = None
        self._frame = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """启动视频流"""
        # 获取流地址
        stream_url = self.client.get_live_stream_url(
            self.device_serial, self.channel_no
        )
        print(f"[Stream] 连接: {stream_url[:50]}...")

        self._cap = cv2.VideoCapture(stream_url)
        if not self._cap.isOpened():
            print("[Stream] 连接失败")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[Stream] 已启动")
        return True

    def _read_loop(self):
        """后台读取循环"""
        while self._running and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """停止视频流"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        print("[Stream] 已停止")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
