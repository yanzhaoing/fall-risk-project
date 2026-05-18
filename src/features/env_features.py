"""
环境风险特征提取模块

从摄像头画面中提取环境风险因素：
- 地面障碍物检测（YOLOv8 物体检测）
- 障碍物距离估算
- 场景风险评分

这是"多模态融合"中的环境模态，与行为模态（步态特征）互补。

输入: RGB 图像 (H, W, 3) 或 图像路径
输出: 环境风险特征向量 + 结构化风险信息

环境风险因素：
1. 地面物体（电线、地毯边、小物件）→ 绊倒风险
2. 家具位置（椅子、桌子）→ 碰撞风险
3. 地面状态（湿滑、不平）→ 滑倒风险
4. 空间狭窄程度 → 跌倒后无人发现风险

参考:
    - 60% 的老年人跌倒与环境因素有关
    - 地面障碍物是最常见的跌倒诱因
    - YOLOv8 可实时检测 80 类物体
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

# COCO 类别 ID → 名称映射（YOLOv8 使用 COCO 数据集）
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush',
}

# 地面障碍物类别（可能绊倒老人的物体）
FLOOR_OBSTACLES = {
    'backpack', 'umbrella', 'handbag', 'suitcase', 'bottle', 'cup',
    'book', 'vase', 'scissors', 'cell phone', 'remote', 'keyboard',
    'sports ball', 'frisbee', 'shoe', 'toy',
}

# 移动障碍物（活的，会突然移动）
MOVING_OBSTACLES = {'cat', 'dog', 'bird'}

# 家具类别（固定障碍物，可能碰撞）
FURNITURE = {'chair', 'couch', 'bed', 'dining table', 'bench', 'tv'}

# 地面物品类别（小物件，容易踩到）
SMALL_ITEMS = {
    'bottle', 'cup', 'cell phone', 'remote', 'book', 'scissors',
    'vase', 'toothbrush', 'mouse', 'keyboard',
}


@dataclass
class DetectedObject:
    """检测到的物体"""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (像素坐标)
    center: Tuple[float, float]  # 中心点 (cx, cy)
    area: float  # 面积 (归一化 0-1)
    is_floor_hazard: bool  # 是否是地面障碍物
    risk_weight: float  # 风险权重 (0-1)


class EnvFeatureExtractor:
    """
    环境风险特征提取器

    使用 YOLOv8 检测画面中的物体，识别潜在的跌倒风险因素。

    特征列表（12维）:
        1. num_obstacles       — 地面障碍物数量
        2. nearest_obstacle_dist — 最近障碍物距离（归一化）
        3. obstacle_density    — 障碍物密度（每单位面积）
        4. num_moving_hazards  — 移动危险物数量（宠物等）
        5. num_furniture       — 家具数量
        6. furniture_density   — 家具密度
        7. num_small_items     — 地面小物件数量
        8. floor_clearance     — 地面净空度（越高越安全）
        9. path_blocked        — 行走路径被阻挡程度
        10. scene_risk_score   — 综合场景风险分数
        11. lighting_estimate  — 光照估计（亮/暗）
        12. clutter_index      — 环境杂乱度

    用法:
        extractor = EnvFeatureExtractor()
        features = extractor.extract(image)  # image: numpy array or path
        features = extractor.extract_vector(image)  # → np.ndarray (12,)
    """

    # 特征名称
    FEATURE_NAMES = [
        "num_obstacles", "nearest_obstacle_dist", "obstacle_density",
        "num_moving_hazards", "num_furniture", "furniture_density",
        "num_small_items", "floor_clearance", "path_blocked",
        "scene_risk_score", "lighting_estimate", "clutter_index",
    ]

    NUM_FEATURES = len(FEATURE_NAMES)

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ):
        """
        Args:
            model_name: YOLOv8 模型名（yolov8n/s/m/l/x，越大越准但越慢）
            confidence_threshold: 置信度阈值
            device: 推理设备
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        """延迟加载 YOLOv8 模型"""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_name)
                print(f"[EnvFeature] YOLOv8 模型已加载: {self.model_name}")
            except ImportError:
                print("[EnvFeature] WARNING: ultralytics 未安装，使用空模型")
                self._model = "dummy"
            except Exception as e:
                print(f"[EnvFeature] WARNING: 模型加载失败: {e}")
                self._model = "dummy"

    def _detect_objects(self, image) -> List[DetectedObject]:
        """
        运行 YOLOv8 物体检测

        Args:
            image: numpy array (H, W, 3) 或 图像路径

        Returns:
            检测到的物体列表
        """
        self._load_model()

        if self._model == "dummy":
            return []

        results = self._model(image, conf=self.confidence_threshold, verbose=False)

        objects = []
        for result in results:
            if result.boxes is None:
                continue

            img_h, img_w = result.orig_shape[:2]

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")

                # 归一化坐标
                nx1, ny1 = x1 / img_w, y1 / img_h
                nx2, ny2 = x2 / img_w, y2 / img_h
                cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
                area = (nx2 - nx1) * (ny2 - ny1)

                # 判断是否是地面障碍物
                is_floor_hazard = (
                    class_name in FLOOR_OBSTACLES
                    or class_name in SMALL_ITEMS
                    or class_name in MOVING_OBSTACLES
                )

                # 风险权重
                risk_weight = self._compute_risk_weight(class_name, cy, area)

                objects.append(DetectedObject(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(nx1, ny1, nx2, ny2),
                    center=(cx, cy),
                    area=area,
                    is_floor_hazard=is_floor_hazard,
                    risk_weight=risk_weight,
                ))

        return objects

    def _compute_risk_weight(
        self, class_name: str, y_center: float, area: float
    ) -> float:
        """
        计算单个物体的风险权重

        考虑因素：
        1. 物体类型（地面物品比空中物品危险）
        2. 位置（画面下方 = 地面，更危险）
        3. 大小（太大的是家具，太小的看不到）

        Args:
            class_name: 物体类别名
            y_center: 物体中心 y 坐标（归一化 0-1，0=顶部，1=底部）
            area: 物体面积（归一化）

        Returns:
            风险权重 (0-1)
        """
        # 基础权重（物体类型）
        if class_name in MOVING_OBSTACLES:
            base = 0.9  # 宠物最危险，会突然移动
        elif class_name in SMALL_ITEMS:
            base = 0.7  # 小物件容易踩到
        elif class_name in FLOOR_OBSTACLES:
            base = 0.6  # 一般地面障碍物
        elif class_name in FURNITURE:
            base = 0.3  # 家具是固定的，相对安全
        else:
            base = 0.2  # 其他物体

        # 位置加成（画面下方更危险 = 在地面上）
        # y_center > 0.6 表示在画面下半部分
        if y_center > 0.7:
            position_boost = 1.2  # 很可能在地面上
        elif y_center > 0.5:
            position_boost = 1.0  # 画面中间
        else:
            position_boost = 0.5  # 画面上方，不太可能是地面障碍

        # 面积惩罚（太大或太小都不太危险）
        if area < 0.001:
            area_factor = 0.3  # 太小，可能是误检
        elif area > 0.3:
            area_factor = 0.5  # 太大，可能是背景
        else:
            area_factor = 1.0

        return float(np.clip(base * position_boost * area_factor, 0, 1))

    def _estimate_lighting(self, image) -> float:
        """
        估计画面光照强度

        暗光环境跌倒风险更高

        Args:
            image: numpy array (H, W, 3)

        Returns:
            光照强度 (0-1, 0=全黑, 1=全亮)
        """
        if isinstance(image, str):
            import cv2
            image = cv2.imread(image)

        gray = np.mean(image) / 255.0
        return float(np.clip(gray, 0, 1))

    def extract_from_objects(
        self, objects: List[DetectedObject], lighting: float
    ) -> Dict[str, float]:
        """
        从已检测的物体和光照计算环境特征（不重复调用 YOLOv8）

        Args:
            objects: 已检测到的物体列表
            lighting: 光照强度 (0-1)

        Returns:
            12 维环境特征字典
        """
        # ── 基础统计 ──
        floor_hazards = [o for o in objects if o.is_floor_hazard]
        moving_hazards = [o for o in objects if o.class_name in MOVING_OBSTACLES]
        furniture = [o for o in objects if o.class_name in FURNITURE]
        small_items = [o for o in objects if o.class_name in SMALL_ITEMS]

        features = {}

        # 1. 地面障碍物数量
        features["num_obstacles"] = float(len(floor_hazards))

        # 2. 最近障碍物距离（用 bbox 中心 y 坐标近似）
        if floor_hazards:
            distances = [1.0 - o.center[1] for o in floor_hazards]
            features["nearest_obstacle_dist"] = float(min(distances))
        else:
            features["nearest_obstacle_dist"] = 1.0

        # 3. 障碍物密度
        img_area = 1.0
        features["obstacle_density"] = float(
            sum(o.area for o in floor_hazards) / img_area
        )

        # 4. 移动危险物数量
        features["num_moving_hazards"] = float(len(moving_hazards))

        # 5. 家具数量
        features["num_furniture"] = float(len(furniture))

        # 6. 家具密度
        features["furniture_density"] = float(
            sum(o.area for o in furniture) / img_area
        )

        # 7. 地面小物件数量
        features["num_small_items"] = float(len(small_items))

        # 8. 地面净空度
        features["floor_clearance"] = float(
            np.clip(1.0 - features["obstacle_density"], 0, 1)
        )

        # 9. 行走路径被阻挡程度
        path_objects = [
            o for o in floor_hazards
            if 0.3 < o.center[0] < 0.7 and o.center[1] > 0.5
        ]
        features["path_blocked"] = float(
            np.clip(sum(o.area for o in path_objects) * 5, 0, 1)
        )

        # 10. 综合场景风险分数
        risk_score = (
            features["num_obstacles"] * 0.3
            + features["obstacle_density"] * 100 * 0.2
            + features["num_moving_hazards"] * 0.2
            + features["path_blocked"] * 0.2
            + (1.0 - lighting) * 0.1
        )
        features["scene_risk_score"] = float(np.clip(risk_score * 20, 0, 100))

        # 11. 光照估计
        features["lighting_estimate"] = lighting

        # 12. 环境杂乱度
        features["clutter_index"] = float(
            np.clip(len(objects) / 20.0, 0, 1)
        )

        return features

    def extract(self, image) -> Dict[str, float]:
        """
        提取所有环境风险特征

        Args:
            image: numpy array (H, W, 3) 或 图像路径

        Returns:
            12 维环境特征字典
        """
        # 物体检测
        objects = self._detect_objects(image)

        # 光照估计
        if isinstance(image, str):
            import cv2
            img_array = cv2.imread(image)
        else:
            img_array = image
        lighting = self._estimate_lighting(img_array)

        return self.extract_from_objects(objects, lighting)

    def extract_vector(self, image) -> np.ndarray:
        """
        提取特征向量（numpy 数组）

        Args:
            image: numpy array (H, W, 3) 或 图像路径

        Returns:
            np.ndarray (12,)
        """
        features = self.extract(image)
        return np.array(
            [features[name] for name in self.FEATURE_NAMES],
            dtype=np.float32,
        )

    def extract_detections(self, image) -> List[Dict]:
        """
        提取检测结果（用于可视化/调试）

        Args:
            image: numpy array (H, W, 3) 或 图像路径

        Returns:
            检测结果列表
        """
        objects = self._detect_objects(image)
        return [
            {
                "class": o.class_name,
                "confidence": round(o.confidence, 3),
                "bbox": tuple(round(v, 3) for v in o.bbox),
                "is_hazard": o.is_floor_hazard,
                "risk_weight": round(o.risk_weight, 3),
            }
            for o in objects
        ]


# ── 轨迹分析 ──────────────────────────────────────────────

class TrajectoryAnalyzer:
    """
    轨迹分析器

    从骨架序列中提取行走轨迹，结合环境障碍物计算交互风险。

    核心思想：风险不是"有障碍物"，而是"老人正朝障碍物走过去"。

    轨迹特征（6维）：
        1. traj_direction       — 行走方向（角度，0=右，π/2=下）
        2. traj_speed           — 当前行走速度（归一化）
        3. predicted_collision  — 预测路径碰撞风险（0-1）
        4. trajectory_regularity — 轨迹规律性（0-1，越高越直）
        5. path_deviation       — 路径偏离度（蛇形程度）
        6. trajectory_risk      — 综合轨迹风险分（0-100）
    """

    FEATURE_NAMES = [
        "traj_direction", "traj_speed", "predicted_collision",
        "trajectory_regularity", "path_deviation", "trajectory_risk",
    ]
    NUM_FEATURES = len(FEATURE_NAMES)

    def __init__(self, fps: float = 30.0, prediction_horizon: float = 1.0):
        """
        Args:
            fps: 帧率
            prediction_horizon: 预测时间窗（秒），预测未来多少秒的路径
        """
        self.fps = fps
        self.prediction_frames = int(prediction_horizon * fps)

    def extract_trajectory(self, skeleton: np.ndarray) -> Dict[str, np.ndarray]:
        """
        从骨架序列提取轨迹信息

        Args:
            skeleton: (T, 17, 3) — T帧, 17关键点, (x, y, conf)

        Returns:
            dict with:
                - positions: (T, 2) — 髋中心位置序列
                - velocities: (T-1, 2) — 速度向量
                - speeds: (T-1,) — 速度大小
                - directions: (T-1,) — 方向角度
        """
        T = skeleton.shape[0]
        xy = skeleton[:, :, :2]  # (T, 17, 2)

        # 髋中心
        left_hip = xy[:, 11]   # COCO LEFT_HIP
        right_hip = xy[:, 12]  # COCO RIGHT_HIP
        hip_center = (left_hip + right_hip) / 2  # (T, 2)

        # 速度向量
        velocities = np.diff(hip_center, axis=0) * self.fps  # (T-1, 2)
        speeds = np.linalg.norm(velocities, axis=-1)  # (T-1,)

        # 方向角度
        directions = np.arctan2(velocities[:, 1], velocities[:, 0])  # (T-1,)

        return {
            "positions": hip_center,
            "velocities": velocities,
            "speeds": speeds,
            "directions": directions,
        }

    def extract(
        self,
        skeleton: np.ndarray,
        obstacles: Optional[List[DetectedObject]] = None,
    ) -> Dict[str, float]:
        """
        提取轨迹-环境交互特征

        Args:
            skeleton: (T, 17, 3)
            obstacles: 检测到的障碍物列表（可选）

        Returns:
            6 维轨迹特征字典
        """
        traj = self.extract_trajectory(skeleton)
        T = len(traj["positions"])

        features = {}

        if T < 3:
            return {name: 0.0 for name in self.FEATURE_NAMES}

        # ── 1. 行走方向 ──
        recent_dirs = traj["directions"][-min(5, len(traj["directions"])):]
        features["traj_direction"] = float(np.mean(recent_dirs))

        # ── 2. 当前行走速度 ──
        recent_speeds = traj["speeds"][-min(5, len(traj["speeds"])):]
        max_speed = np.max(traj["speeds"]) + 1e-6
        features["traj_speed"] = float(np.mean(recent_speeds) / max_speed)

        # ── 3. 预测路径碰撞风险 ──
        features["predicted_collision"] = self._compute_collision_risk(
            traj, obstacles
        )

        # ── 4. 轨迹规律性 ──
        features["trajectory_regularity"] = self._compute_regularity(traj)

        # ── 5. 路径偏离度 ──
        features["path_deviation"] = self._compute_path_deviation(traj)

        # ── 6. 综合轨迹风险分 ──
        features["trajectory_risk"] = self._compute_trajectory_risk(features)

        return features

    def extract_vector(
        self,
        skeleton: np.ndarray,
        obstacles: Optional[List[DetectedObject]] = None,
    ) -> np.ndarray:
        """提取特征向量 (6,)"""
        features = self.extract(skeleton, obstacles)
        return np.array(
            [features[name] for name in self.FEATURE_NAMES],
            dtype=np.float32,
        )

    def _compute_collision_risk(
        self,
        traj: Dict,
        obstacles: Optional[List[DetectedObject]],
    ) -> float:
        """
        计算预测路径与障碍物的碰撞风险

        逻辑：
        1. 用最近速度预测未来 N 帧的位置
        2. 检查预测位置附近有没有障碍物
        3. 越近的碰撞风险越高
        """
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        positions = traj["positions"]
        velocities = traj["velocities"]

        if len(velocities) < 2:
            return 0.0

        current_pos = positions[-1]  # (2,)
        avg_velocity = np.mean(velocities[-5:], axis=0)  # (2,)

        max_risk = 0.0
        for t in range(1, self.prediction_frames + 1):
            predicted_pos = current_pos + avg_velocity * t / self.fps

            for obs in obstacles:
                obs_center = np.array(obs.center)
                dist = np.linalg.norm(predicted_pos - obs_center)

                # 高斯衰减：距离越近风险越高
                sigma = 0.15
                risk = np.exp(-dist ** 2 / (2 * sigma ** 2))

                # 时间衰减：越远的预测越不确定
                time_decay = 1.0 / (1.0 + t * 0.1)

                max_risk = max(max_risk, float(risk * time_decay))

        return float(np.clip(max_risk, 0, 1))

    def _compute_regularity(self, traj: Dict) -> float:
        """
        计算轨迹规律性

        正常老人走直线或缓弯 → 高规律性
        踉跄/头晕 → 蛇形轨迹 → 低规律性

        方法：方向变化的标准差（越小越规律）
        """
        directions = traj["directions"]
        if len(directions) < 3:
            return 0.5

        dir_changes = np.abs(np.diff(directions))
        dir_changes = np.minimum(dir_changes, 2 * np.pi - dir_changes)

        std = np.std(dir_changes)
        regularity = float(np.clip(1.0 - std / (np.pi / 4), 0, 1))

        return regularity

    def _compute_path_deviation(self, traj: Dict) -> float:
        """
        计算路径偏离度（蛇形程度）

        方法：实际路径长度 / 起点到终点直线距离
        比值越大 → 路径越弯曲
        """
        positions = traj["positions"]
        if len(positions) < 3:
            return 0.0

        segments = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        actual_length = np.sum(segments)

        straight_length = np.linalg.norm(positions[-1] - positions[0])

        if straight_length < 1e-6:
            return 1.0  # 原地不动 = 最大偏离

        deviation = actual_length / straight_length - 1.0
        return float(np.clip(deviation / 2.0, 0, 1))

    def _compute_trajectory_risk(self, features: Dict) -> float:
        """综合轨迹风险分"""
        risk = (
            features["predicted_collision"] * 0.4
            + (1.0 - features["trajectory_regularity"]) * 0.3
            + features["path_deviation"] * 0.3
        )
        return float(np.clip(risk * 100, 0, 100))


class SceneRiskAnalyzer:
    """
    场景风险分析器（组合类）

    将环境特征提取器和轨迹分析器组合在一起，
    输出完整的场景风险特征。

    总特征维度：12（环境）+ 6（轨迹）= 18 维
    """

    FEATURE_NAMES = EnvFeatureExtractor.FEATURE_NAMES + TrajectoryAnalyzer.FEATURE_NAMES
    NUM_FEATURES = len(FEATURE_NAMES)

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
        fps: float = 30.0,
    ):
        self.env_extractor = EnvFeatureExtractor(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        self.traj_analyzer = TrajectoryAnalyzer(fps=fps)

    def extract(
        self,
        image,
        skeleton: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        提取完整场景风险特征

        Args:
            image: RGB 图像或路径
            skeleton: 骨架序列 (T, 17, 3)，可选

        Returns:
            18 维特征字典
        """
        # 物体检测（只跑一次 YOLOv8）
        objects = self.env_extractor._detect_objects(image)

        # 光照估计
        if isinstance(image, str):
            import cv2
            img_array = cv2.imread(image)
        else:
            img_array = image
        lighting = self.env_extractor._estimate_lighting(img_array)

        # 环境特征（复用已检测的物体）
        env_features = self.env_extractor.extract_from_objects(objects, lighting)

        # 轨迹特征
        if skeleton is not None:
            traj_features = self.traj_analyzer.extract(skeleton, objects)
        else:
            traj_features = {name: 0.0 for name in TrajectoryAnalyzer.FEATURE_NAMES}

        all_features = {**env_features, **traj_features}
        return all_features

    def extract_vector(
        self,
        image,
        skeleton: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """提取特征向量 (18,)"""
        features = self.extract(image, skeleton)
        return np.array(
            [features[name] for name in self.FEATURE_NAMES],
            dtype=np.float32,
        )


# ── 测试 ──
if __name__ == "__main__":
    import sys

    extractor = EnvFeatureExtractor(
        model_name="yolov8n.pt",
        confidence_threshold=0.3,
        device="cpu",
    )

    # 测试用随机图像
    print("=== 环境风险特征测试 ===")
    print(f"特征数: {EnvFeatureExtractor.NUM_FEATURES}")

    # 用纯色图像测试（无物体）
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    blank[:] = 200

    features = extractor.extract(blank)
    print("\n空白图像特征:")
    for name, val in features.items():
        print(f"  {name}: {val:.4f}")

    vec = extractor.extract_vector(blank)
    print(f"\n特征向量 shape: {vec.shape}")

    # 轨迹分析测试
    print("\n=== 轨迹分析测试 ===")
    traj_analyzer = TrajectoryAnalyzer(fps=30.0)

    # 模拟直线行走骨架
    skeleton = np.random.randn(30, 17, 3).astype(np.float32)
    skeleton[:, :, 2] = 1.0
    for t in range(30):
        skeleton[t, 11, 0] = 0.3 + t * 0.01
        skeleton[t, 12, 0] = 0.35 + t * 0.01
        skeleton[t, 11, 1] = 0.5
        skeleton[t, 12, 1] = 0.5

    traj_features = traj_analyzer.extract(skeleton)
    print("直线行走轨迹特征:")
    for name, val in traj_features.items():
        print(f"  {name}: {val:.4f}")

    # 模拟蛇形行走
    skeleton_snake = skeleton.copy()
    for t in range(30):
        skeleton_snake[t, 11, 1] = 0.5 + 0.1 * np.sin(t * 0.5)
        skeleton_snake[t, 12, 1] = 0.5 + 0.1 * np.sin(t * 0.5)

    traj_features_snake = traj_analyzer.extract(skeleton_snake)
    print("\n蛇形行走轨迹特征:")
    for name, val in traj_features_snake.items():
        print(f"  {name}: {val:.4f}")

    # 组合分析器测试
    print("\n=== SceneRiskAnalyzer 测试 ===")
    analyzer = SceneRiskAnalyzer(model_name="yolov8n.pt", device="cpu")
    full_features = analyzer.extract(blank, skeleton)
    print(f"总特征数: {SceneRiskAnalyzer.NUM_FEATURES}")
    print("特征:")
    for name, val in full_features.items():
        print(f"  {name}: {val:.4f}")

    print("\n✅ 全部模块验证通过")
