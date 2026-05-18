"""
萤石开放平台 API 客户端

封装萤石平台的核心 API:
- 获取 access_token
- 人体检测 API
- 姿态分析 API
- 设备管理

API 文档: https://open.ys7.com
"""
import requests
import time
from typing import Dict, List, Optional

from config.settings import CFG_EZVIZ


class EZVIZClient:
    """
    萤石开放平台 API 客户端

    Args:
        app_key: 应用 Key
        app_secret: 应用 Secret
    """

    def __init__(
        self,
        app_key: str = CFG_EZVIZ.APP_KEY,
        app_secret: str = CFG_EZVIZ.APP_SECRET,
    ):
        self.app_key = app_key
        self.app_secret = app_secret
        self.access_token = CFG_EZVIZ.ACCESS_TOKEN
        self.token_expires_at = 0

    def _ensure_token(self):
        """确保 access_token 有效"""
        if self.access_token and time.time() < self.token_expires_at:
            return

        url = f"{CFG_EZVIZ.API_BASE}/api/lapp/token/get"
        resp = requests.post(url, data={
            "appKey": self.app_key,
            "appSecret": self.app_secret,
        })
        data = resp.json()

        if data.get("code") == "200":
            self.access_token = data["data"]["accessToken"]
            self.token_expires_at = time.time() + data["data"]["expireTime"] - 300
            print("[EZVIZ] Token 获取成功")
        else:
            raise RuntimeError(f"EZVIZ Token 获取失败: {data}")

    def _headers(self) -> Dict:
        """构建请求头"""
        self._ensure_token()
        return {"Content-Type": "application/x-www-form-urlencoded"}

    def _params(self, extra: Dict = None) -> Dict:
        """构建请求参数"""
        self._ensure_token()
        p = {"accessToken": self.access_token}
        if extra:
            p.update(extra)
        return p

    def get_device_list(self) -> List[Dict]:
        """获取设备列表"""
        url = f"{CFG_EZVIZ.API_BASE}/api/lapp/device/list"
        resp = requests.post(url, params=self._params())
        data = resp.json()
        if data.get("code") == "200":
            return data.get("data", {}).get("devices", [])
        return []

    def body_detect(
        self,
        image_data: bytes,
    ) -> Dict:
        """
        调用人体检测 API

        Args:
            image_data: 图片二进制数据

        Returns:
            检测结果
        """
        url = CFG_EZVIZ.BODY_DETECT_URL
        files = {"image": ("frame.jpg", image_data, "image/jpeg")}
        resp = requests.post(
            url,
            params=self._params(),
            files=files,
        )
        return resp.json()

    def pose_analysis(
        self,
        image_data: bytes,
    ) -> Dict:
        """
        调用姿态分析 API

        Args:
            image_data: 图片二进制数据

        Returns:
            姿态分析结果
        """
        url = CFG_EZVIZ.POSE_ANALYSIS_URL
        files = {"image": ("frame.jpg", image_data, "image/jpeg")}
        resp = requests.post(
            url,
            params=self._params(),
            files=files,
        )
        return resp.json()

    def get_live_stream_url(
        self,
        device_serial: str,
        channel_no: int = 1,
    ) -> str:
        """
        获取直播流地址

        Args:
            device_serial: 设备序列号
            channel_no: 通道号

        Returns:
            RTSP/RTMP 流地址
        """
        url = f"{CFG_EZVIZ.API_BASE}/api/lapp/live/address/get"
        resp = requests.post(url, params=self._params({
            "deviceSerial": device_serial,
            "channelNo": channel_no,
            "protocol": 2,  # RTSP
        }))
        data = resp.json()
        if data.get("code") == "200":
            return data["data"]["url"]
        raise RuntimeError(f"获取直播流失败: {data}")
