import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
import cv2
import time
import logging
import math
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet34, ResNet34_Weights
from torch.optim.lr_scheduler import OneCycleLR
from einops import rearrange, repeat
from collections import defaultdict
import hashlib
import json
from typing import List, Dict, Optional
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError

# -------------------- 配置文件路径 --------------------
MAP_ROOT = "./map_features"
DATA_ROOT = "./data"
VEHICLE_PARAMS = {
    'max_steer': math.radians(35),
    'max_accel': 4.0,
    'max_decel': 6.0,
    'speed_limit': 20.0,  # 72 km/h
    'max_lat_acc': 3.0  # 新增最大横向加速度参数
}


# -------------------- 修改后的日志配置 --------------------
class HTTPFilter(logging.Filter):
    """过滤HTTP请求日志"""

    def filter(self, record):
        return "HTTP Request" not in record.getMessage()


# 配置日志记录
log_directory = os.path.expanduser('~/carla_logs')
os.makedirs(log_directory, exist_ok=True)

# 禁用第三方库的verbose日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# 创建处理器并添加过滤器
handlers = []
file_handler = logging.FileHandler(
    os.path.join(log_directory, 'processing.log'),
    encoding='utf-8'
)
file_handler.addFilter(HTTPFilter())
handlers.append(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.addFilter(HTTPFilter())
handlers.append(stream_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=handlers  # 使用配置好的处理器列表
)
logger = logging.getLogger(__name__)
logger.addFilter(HTTPFilter())


# -------------------- 数据集类 --------------------
class EnhancedAccidentDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, map_root=MAP_ROOT, split="train",
                 input_len=20, output_len=30, map_size=256, mode='train',
                 augment_prob=None, noise_scale=None, scalers=None):
        self.data_dir = os.path.join(data_root, split)
        self.map_root = map_root
        self.vector_map_dir = map_root
        self.input_len = input_len
        self.output_len = output_len
        self.map_size = map_size
        self.mode = mode
        self.min_traj_length = input_len + output_len
        self.max_speed = 25.0  # 新增速度过滤
        self.min_speed = 0.5  # 最小有效速度

        # 角色映射定义
        self.role_mapping = {
            0: "ego_vehicle",
            1: "ego_vehicle_behind",
            2: "other_vehicle",
            3: "other_vehicle_behind"
        }

        self._validate_paths()
        self.file_list = self._scan_files_v2()
        self._log_dataset_stats()

        self.scalers = self._init_scalers() if scalers is None else scalers

        self.augment_prob = augment_prob if augment_prob is not None else (0.8 if mode == 'train' else 0.0)
        self.noise_scale = noise_scale if noise_scale is not None else torch.tensor([0.2, 0.2, 0.1, 0.3, 0.3])

    def _validate_paths(self):
        required_dirs = [self.data_dir, self.map_root]
        for path in required_dirs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"关键路径不存在: {path}")

    def _scan_files_v2(self):
        scenarios = []
        pattern = re.compile(r"type\d+_subtype\d+_(accident|normal)")

        # 遍历场景类型目录
        scenario_type_dirs = [d for d in os.listdir(self.data_dir)
                              if pattern.match(d) and os.path.isdir(os.path.join(self.data_dir, d))]

        if not scenario_type_dirs:
            logger.warning(f"在 {self.data_dir} 中未找到有效的场景类型目录")
            return scenarios

        for scenario_type in scenario_type_dirs:
            scenario_dir = os.path.join(self.data_dir, scenario_type)
            meta_dir = os.path.join(scenario_dir, "meta")

            if not os.path.exists(meta_dir):
                logger.warning(f"元数据目录不存在: {meta_dir}")
                continue

            # 收集所有元数据文件
            meta_files = [f for f in os.listdir(meta_dir) if f.endswith(".txt")]
            if not meta_files:
                logger.warning(f"在 {meta_dir} 中未找到元数据文件")
                continue

            for meta_file in meta_files:
                scene_id = os.path.splitext(meta_file)[0]

                # 解析场景信息并检查是否为Town07
                scenario_info = self._parse_scenario_info(scenario_dir, scene_id)
                if scenario_info['town'] == "Town07":
                    logger.info(f"跳过Town07场景: {scene_id}")
                    continue  # 跳过Town07场景

                scenario_data = {
                    "meta": os.path.join(meta_dir, meta_file),
                    "calib": {},
                    "lidar": {},
                    "labels": {},  # 只存储ego_vehicle的标签
                    "scenario_info": scenario_info
                }

                # 收集标定数据
                calib_dir = os.path.join(scenario_dir, "ego_vehicle", "calib", scene_id)
                if os.path.exists(calib_dir):
                    calib_files = [f for f in os.listdir(calib_dir) if f.endswith(".pkl")]
                    for calib_file in calib_files:
                        frame_num = int(calib_file.split('_')[-1].split('.')[0])
                        scenario_data["calib"][frame_num] = os.path.join(calib_dir, calib_file)

                # 收集LiDAR数据
                lidar_dir = os.path.join(scenario_dir, "ego_vehicle", "lidar01", scene_id)
                if os.path.exists(lidar_dir):
                    lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith(".npz")]
                    for lidar_file in lidar_files:
                        frame_num = int(lidar_file.split('_')[-1].split('.')[0])
                        scenario_data["lidar"][frame_num] = os.path.join(lidar_dir, lidar_file)

                # 只收集ego_vehicle的标签文件
                role_label_dir = os.path.join(scenario_dir, "ego_vehicle", "label", scene_id)
                if os.path.exists(role_label_dir):
                    label_files = [f for f in os.listdir(role_label_dir) if f.endswith(".txt")]
                    frame_labels = {}
                    for label_file in label_files:
                        try:
                            frame_num = int(label_file.split('_')[-1].split('.')[0])
                            frame_labels[frame_num] = os.path.join(role_label_dir, label_file)
                        except:
                            continue
                    scenario_data["labels"] = frame_labels

                # 确保有足够的数据帧
                if scenario_data["calib"] and scenario_data["lidar"] and scenario_data["labels"]:
                    scenarios.append(scenario_data)
                else:
                    logger.warning(f"场景 {scene_id} 数据不完整: calib={bool(scenario_data['calib'])}, "
                                   f"lidar={bool(scenario_data['lidar'])}, labels={bool(scenario_data['labels'])}")

        logger.info(f"共找到 {len(scenarios)} 个有效场景")
        return scenarios

    def _parse_meta_file(self, meta_path):
        """改进的元数据解析，更健壮地提取agents id"""
        meta_info = {"agent_ids": []}
        try:
            with open(meta_path, 'r') as f:
                content = f.read()

                # 更健壮的正则表达式匹配agents id
                id_pattern = r"agents\s+id:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
                id_match = re.search(id_pattern, content, re.IGNORECASE)

                if id_match:
                    ids = [int(id_match.group(1)), int(id_match.group(2)),
                           int(id_match.group(3)), int(id_match.group(4))]
                    meta_info["agent_ids"] = ids
                else:
                    # 尝试备用匹配模式
                    alt_pattern = r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
                    alt_match = re.search(alt_pattern, content)
                    if alt_match:
                        ids = [int(alt_match.group(1)), int(alt_match.group(2)),
                               int(alt_match.group(3)), int(alt_match.group(4))]
                        meta_info["agent_ids"] = ids
                    else:
                        logger.warning(f"无法在元数据文件中解析agent_ids: {meta_path}")
                        meta_info["agent_ids"] = [-1, -1, -1, -1]

                # 解析道路类型
                road_match = re.search(r"road_type:\s*(.+)", content, re.IGNORECASE)
                meta_info["road_type"] = road_match.group(1).strip() if road_match else "unknown"

        except Exception as e:
            logger.error(f"解析元数据文件失败: {meta_path}, 错误: {str(e)}")
            meta_info = {"agent_ids": [-1, -1, -1, -1], "road_type": "unknown"}

        return meta_info

    def _is_valid_trajectory(self, traj):
        """改进的数据清洗逻辑"""
        if len(traj) < self.min_traj_length:
            return False
        speeds = np.linalg.norm(traj[:, 3:5], axis=1)
        if np.any(speeds > self.max_speed):
            return False
        return True

    def _parse_scenario_info(self, path, scene_id):
        """从路径和场景ID中解析场景信息"""
        # 从场景ID中提取信息
        pattern = r"Town(\d+)_type(\d+)_subtype(\d+)_scenario(\d+)"
        match = re.search(pattern, scene_id)
        scenario_type = "accident" if "accident" in os.path.basename(path) else "normal"

        if match:
            return {
                'town': f"Town{match.group(1)}",
                'type_id': int(match.group(2)),
                'subtype_id': int(match.group(3)),
                'scenario_id': int(match.group(4)),
                'scenario_type': scenario_type
            }

        # 尝试从路径中提取信息
        base_name = os.path.basename(path)
        match = re.search(r"type(\d+)_subtype(\d+)_(accident|normal)", base_name)
        if match:
            return {
                'town': 'Unknown',
                'type_id': int(match.group(1)),
                'subtype_id': int(match.group(2)),
                'scenario_id': 0,
                'scenario_type': "accident" if "accident" in base_name else "normal"
            }

        return {
            'town': 'Unknown',
            'type_id': 0,
            'subtype_id': 0,
            'scenario_id': 0,
            'scenario_type': scenario_type
        }

    def _log_dataset_stats(self):
        stats = defaultdict(int)
        for item in self.file_list:
            info = item['scenario_info']
            key = f"{info['town']}_type{info['type_id']}"
            stats[key] += 1
        logger.info("数据集统计:")
        for k, v in stats.items():
            logger.info(f"• {k}: {v} samples")
        logger.info(f"总样本数: {len(self.file_list)}")


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        max_retries = 5
        original_idx = idx
        for attempt in range(max_retries):
            try:
                scenario = self.file_list[idx]
                meta_info = self._parse_meta_file(scenario["meta"])

                # 从元数据中提取车辆ID
                agent_ids = meta_info["agent_ids"]
                ego_id = agent_ids[0]
                ego_behind_id = agent_ids[1]
                other_id = agent_ids[2]
                other_behind_id = agent_ids[3]

                logger.debug(f"车辆ID: ego={ego_id}, ego_behind={ego_behind_id}, "
                             f"other={other_id}, other_behind={other_behind_id}")

                # 获取所有可用帧
                calib_frames = set(scenario["calib"].keys())
                lidar_frames = set(scenario["lidar"].keys())
                label_frames = set(scenario["labels"].keys())
                all_frames = sorted(calib_frames & lidar_frames & label_frames)

                if len(all_frames) < self.min_traj_length:
                    raise ValueError(f"场景只有 {len(all_frames)} 帧，少于最小要求 {self.min_traj_length}")

                # 初始化轨迹容器
                traj_data = {
                    "ego": {"id": ego_id, "points": []},
                    "ego_behind": {"id": ego_behind_id, "points": []},
                    "other": {"id": other_id, "points": []},
                    "other_behind": {"id": other_behind_id, "points": []}
                }

                # 按帧顺序处理数据
                for frame_idx in all_frames:
                    # 加载标定参数
                    calib_path = scenario["calib"].get(frame_idx)
                    calib = {'lidar_to_cam': None, 'cam_to_world': None}
                    if calib_path and os.path.exists(calib_path):
                        try:
                            with open(calib_path, 'rb') as f:
                                calib = pickle.load(f)
                        except Exception as e:
                            logger.warning(f"加载标定文件失败: {calib_path}, 错误: {str(e)}")

                    # 加载标签文件
                    label_path = scenario["labels"].get(frame_idx)
                    if not label_path or not os.path.exists(label_path):
                        continue

                    try:
                        with open(label_path, 'r') as f:
                            # 读取传感器位姿 (第一行)
                            sensor_line = f.readline().strip().split()
                            if len(sensor_line) < 2:
                                continue

                            try:
                                timestamp = float(sensor_line[0])
                                sensor_yaw = float(sensor_line[1])
                                sensor_pose = (0, 0, sensor_yaw)  # 传感器位置未知，暂时设为(0,0)
                            except Exception as e:
                                logger.warning(f"解析传感器位姿失败: {label_path}, 错误: {str(e)}")
                                sensor_pose = (0, 0, 0.0)

                            # 处理所有物体
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) < 13:
                                    continue

                                try:
                                    obj_id = int(parts[10])
                                except:
                                    continue

                                # 提取物体数据
                                try:
                                    local_x = float(parts[1])
                                    local_y = float(parts[2])
                                    yaw = float(parts[7])
                                    vel_x = float(parts[8])
                                    vel_y = float(parts[9])

                                    # 匹配车辆ID
                                    role = None
                                    if obj_id == ego_id or obj_id == -100:  # ego车辆的特殊ID
                                        role = "ego"
                                    elif obj_id == ego_behind_id:
                                        role = "ego_behind"
                                    elif obj_id == other_id:
                                        role = "other"
                                    elif obj_id == other_behind_id:
                                        role = "other_behind"

                                    if role:
                                        # 坐标转换: 局部到全局
                                        global_pos = self._local_to_global(
                                            [local_x, local_y],
                                            sensor_pose,
                                            calib
                                        )

                                        # 速度方向转换
                                        theta = sensor_pose[2]  # 使用传感器航向
                                        cos_theta = np.cos(theta)
                                        sin_theta = np.sin(theta)
                                        global_vel_x = vel_x * cos_theta - vel_y * sin_theta
                                        global_vel_y = vel_x * sin_theta + vel_y * cos_theta

                                        # 航向角转换 (局部到全局)
                                        global_yaw = theta + yaw
                                        # 归一化到[-π, π]
                                        global_yaw = (global_yaw + math.pi) % (2 * math.pi) - math.pi

                                        traj_point = [
                                            global_pos[0],  # x
                                            global_pos[1],  # y
                                            global_yaw,  # yaw
                                            global_vel_x,  # vel_x
                                            global_vel_y  # vel_y
                                        ]

                                        # 添加到轨迹
                                        traj_data[role]["points"].append(traj_point)
                                except Exception as e:
                                    logger.warning(f"轨迹点处理失败: {str(e)}")
                                    continue
                    except Exception as e:
                        logger.error(f"处理标签文件失败: {label_path}, 错误: {str(e)}")

                # 处理轨迹数据
                processed_trajs = {}
                for role, data in traj_data.items():
                    if data["points"]:
                        arr = np.array(data["points"])
                        processed_traj = self._process_trajectory(arr)
                        processed_trajs[role] = {
                            "id": data["id"],
                            "traj": processed_traj
                        }
                    else:
                        logger.warning(f"角色 {role} 的轨迹为空")
                        processed_trajs[role] = {
                            "id": data["id"],
                            "traj": np.zeros((self.input_len + self.output_len, 5))
                        }

                # 提取主车轨迹
                ego_data = processed_trajs.get("ego", {})
                ego_traj = ego_data.get("traj", [])

                if len(ego_traj) == 0:
                    raise ValueError("主车轨迹为空，无效样本")

                # 分割输入和输出轨迹
                if len(ego_traj) < self.input_len + self.output_len:
                    pad_len = self.input_len + self.output_len - len(ego_traj)
                    ego_traj = np.vstack([ego_traj, np.zeros((pad_len, 5))])

                input_traj = ego_traj[:self.input_len]
                target_traj = ego_traj[self.input_len:self.input_len + self.output_len]

                # 确保轨迹长度正确
                if len(input_traj) < self.input_len:
                    pad = np.zeros((self.input_len - len(input_traj), 5))
                    input_traj = np.vstack([input_traj, pad])

                if len(target_traj) < self.output_len:
                    pad = np.zeros((self.output_len - len(target_traj), 5))
                    target_traj = np.vstack([target_traj, pad])

                # 收集其他车辆轨迹
                other_trajs = []
                for role in ["ego_behind", "other", "other_behind"]:
                    if role in processed_trajs:
                        other_trajs.append({
                            "id": processed_trajs[role]["id"],
                            "traj": processed_trajs[role]["traj"],
                            "role": role
                        })

                # 加载地图数据
                map_data = self._load_map(scenario["scenario_info"]["town"])

                # 生成意图标签
                yaw_changes = np.diff(target_traj[:, 2])
                avg_yaw_change = np.mean(yaw_changes) if len(yaw_changes) > 0 else 0
                lateral_intention = 0 if avg_yaw_change > 0.1 else 2 if avg_yaw_change < -0.1 else 1

                initial_speed = np.linalg.norm(target_traj[0, 3:5]) if len(target_traj) > 0 else 0
                final_speed = np.linalg.norm(target_traj[-1, 3:5]) if len(target_traj) > 0 else 0
                delta_speed = final_speed - initial_speed
                longitudinal_intention = 0 if delta_speed > 0.5 else 2 if delta_speed < -0.5 else 1

                # 数据增强
                input_tensor = torch.FloatTensor(input_traj)
                if self.mode == 'train':
                    input_tensor = self._augment_trajectory(input_tensor)

                # 计算有效长度
                input_valid_len = min(len(ego_traj), self.input_len)
                output_valid_len = min(len(ego_traj) - self.input_len, self.output_len)
                if output_valid_len < 0:
                    output_valid_len = 0

                # 改进道路类型编码
                road_type = self._encode_road_type(meta_info.get("road_type", "unknown"))

                return {
                    'input_traj': input_tensor,
                    'target_traj': torch.FloatTensor(target_traj[..., :2]),
                    'raw_targets': torch.FloatTensor(ego_traj[self.input_len:, :2]),
                    'velocity': torch.FloatTensor(ego_traj[self.input_len:, 3:5]),
                    'vector_map': {
                        'polylines': torch.FloatTensor(map_data['polylines']),
                        'poly_meta': torch.FloatTensor(map_data['poly_meta'])
                    },
                    'scenario_type': 1 if scenario["scenario_info"]['scenario_type'] == "accident" else 0,
                    'intention_label': torch.LongTensor([lateral_intention, longitudinal_intention]),
                    'input_valid_len': input_valid_len,
                    'output_valid_len': output_valid_len,
                    'ego_id': ego_id,
                    'other_trajs': other_trajs,
                    'road_type': road_type
                }
            except Exception as e:
                logger.error(f"加载样本 {idx} 失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}", exc_info=True)
                idx = (idx + 1) % len(self.file_list)  # 尝试下一个样本

        raise RuntimeError(f"多次重试后仍无法加载原始索引 {original_idx} 的样本")

    def _local_to_global(self, local_point, sensor_pose, calib):
        """更健壮的坐标转换函数，使用传感器航向"""
        try:
            x_local, y_local = local_point
            sensor_x, sensor_y, sensor_yaw = sensor_pose

            # 应用传感器航向旋转
            cos_yaw = math.cos(sensor_yaw)
            sin_yaw = math.sin(sensor_yaw)

            # 旋转和平移
            x_global = x_local * cos_yaw - y_local * sin_yaw + sensor_x
            y_global = x_local * sin_yaw + y_local * cos_yaw + sensor_y

            return np.array([x_global, y_global])
        except Exception as e:
            logger.error(f"坐标转换失败: {str(e)}")
            return np.array(local_point)  # 返回原始点作为回退

    def _process_trajectory(self, traj):
        """更健壮的轨迹处理"""
        if len(traj) == 0:
            return np.zeros((self.input_len + self.output_len, 5))

        # 速度过滤 - 确保不会因单个异常点导致整个轨迹被丢弃
        valid_mask = np.ones(len(traj), dtype=bool)
        if len(traj) > 5:  # 只在有足够点时进行过滤
            speeds = np.linalg.norm(traj[:, 3:5], axis=1)
            valid_mask = (speeds > self.min_speed) & (speeds < self.max_speed)

            # 如果过滤后点太少，保留原始轨迹
            if np.sum(valid_mask) < 5:
                valid_mask = np.ones(len(traj), dtype=bool)

        filtered = traj[valid_mask]

        if len(filtered) == 0:
            return np.zeros((self.input_len + self.output_len, 5))

        try:
            # 标准化（位置和速度）
            if hasattr(self.scalers['position'], 'transform'):
                filtered[:, 0:2] = self.scalers['position'].transform(filtered[:, 0:2])
            if hasattr(self.scalers['velocity'], 'transform'):
                filtered[:, 3:5] = self.scalers['velocity'].transform(filtered[:, 3:5])
        except Exception as e:
            logger.error(f"轨迹标准化失败: {str(e)}")

        # 如果轨迹长度不足，进行线性插值
        if len(filtered) < self.input_len + self.output_len:
            return self._interpolate_trajectory(filtered)

        return filtered

    def _interpolate_trajectory(self, traj):
        """轨迹插值函数"""
        original_len = len(traj)
        if original_len == 0:
            return np.zeros((self.input_len + self.output_len, 5))

        # 创建时间索引
        orig_indices = np.arange(original_len)
        target_indices = np.linspace(0, original_len - 1, self.input_len + self.output_len)

        # 对每个维度进行插值
        interpolated = np.zeros((len(target_indices), 5))
        for i in range(5):
            interpolated[:, i] = np.interp(target_indices, orig_indices, traj[:, i])

        return interpolated

    def _process_velocity(self, velocity, target_len):
        """处理速度序列确保长度一致"""
        if velocity.shape[0] > target_len:
            return velocity[:target_len]
        pad = np.zeros((target_len - velocity.shape[0], 2))
        return np.vstack([velocity, pad])

    def _init_scalers(self):
        """利用所有文件中的 ego 轨迹拟合 scaler"""
        pos_list = []
        vel_list = []

        for scenario in self.file_list:
            try:
                meta_info = self._parse_meta_file(scenario["meta"])

                # 修复5: 检查agent_ids长度
                if not meta_info["agent_ids"] or len(meta_info["agent_ids"]) < 1:
                    logger.warning(f"元数据中没有agent_ids: {scenario['meta']}")
                    continue

                ego_id = meta_info["agent_ids"][0]

                # 尝试加载主车轨迹
                if "ego_vehicle" in scenario["labels"]:
                    ego_labels = scenario["labels"]["ego_vehicle"]
                    # 收集轨迹点
                    traj_points = []
                    for frame_idx, label_path in ego_labels.items():
                        if not os.path.exists(label_path):
                            continue

                        with open(label_path) as f:
                            # 跳过第一行（传感器信息）
                            f.readline()

                            for line in f:
                                parts = line.strip().split()
                                if len(parts) < 13:
                                    continue

                                try:
                                    obj_id = int(parts[10])
                                except:
                                    continue

                                if obj_id == ego_id or obj_id == -100:
                                    try:
                                        traj_points.append([
                                            float(parts[1]),  # x
                                            float(parts[2]),  # y
                                            float(parts[7]),  # yaw
                                            float(parts[8]),  # vel_x
                                            float(parts[9])  # vel_y
                                        ])
                                    except:
                                        continue

                    if traj_points and len(traj_points) >= self.min_traj_length:
                        traj = np.array(traj_points)
                        pos_list.append(traj[:, :2])
                        vel_list.append(traj[:, 3:])
            except Exception as e:
                logger.warning(f"初始化scaler失败: {str(e)}")
                continue

        # 创建并拟合scaler
        pos_scaler = StandardScaler()
        vel_scaler = StandardScaler()

        if pos_list:
            pos_scaler.fit(np.vstack(pos_list))
        else:
            # 使用默认值避免空scaler
            pos_scaler.mean_ = np.array([0.0, 0.0])
            pos_scaler.scale_ = np.array([1.0, 1.0])
            logger.warning("位置scaler使用默认值，未找到有效轨迹")

        if vel_list:
            vel_scaler.fit(np.vstack(vel_list))
        else:
            # 使用默认值避免空scaler
            vel_scaler.mean_ = np.array([0.0, 0.0])
            vel_scaler.scale_ = np.array([1.0, 1.0])
            logger.warning("速度scaler使用默认值，未找到有效轨迹")

        return {'position': pos_scaler, 'velocity': vel_scaler}

    def _load_map(self, town):
        # 跳过Town07地图加载
        if town == "Town07":
            logger.info(f"跳过Town07地图加载")
            return {
                'polylines': np.zeros((200, 50, 6), dtype=np.float32),
                'poly_meta': np.zeros((200, 4), dtype=np.float32)
            }

        map_path = f"{self.map_root}/{town}.npz"
        try:
            if os.path.exists(map_path):
                data = np.load(map_path)

                # 修复6: 处理地图文件结构问题
                # 检查文件是否包含所需的数组
                if 'polylines' in data and 'poly_meta' in data:
                    return {
                        'polylines': data['polylines'].astype(np.float32),
                        'poly_meta': data['poly_meta'].astype(np.float32)
                    }
                else:
                    # 尝试其他可能的键名
                    possible_keys = {
                        'polylines': ['polylines', 'vectors', 'lines'],
                        'poly_meta': ['poly_meta', 'meta', 'metadata']
                    }

                    found = False
                    for poly_key in possible_keys['polylines']:
                        for meta_key in possible_keys['poly_meta']:
                            if poly_key in data and meta_key in data:
                                logger.info(f"使用替代键名加载地图: {poly_key}, {meta_key}")
                                return {
                                    'polylines': data[poly_key].astype(np.float32),
                                    'poly_meta': data[meta_key].astype(np.float32)
                                }

                    logger.error(f"地图文件 {map_path} 中缺少所需的数组")
        except Exception as e:
            logger.error(f"加载地图失败: {map_path}, 错误: {str(e)}")

        # 返回空地图作为占位符
        logger.warning(f"为{town}使用空地图占位符")
        return {
            'polylines': np.zeros((200, 50, 6), dtype=np.float32),
            'poly_meta': np.zeros((200, 4), dtype=np.float32)
        }

    def _encode_road_type(self, road_type):
        """道路类型编码"""
        # 修复7: 改进道路类型编码逻辑
        if not isinstance(road_type, str):
            return 4  # unknown

        road_type = road_type.lower()

        if 'junction' in road_type or 'cross' in road_type:
            if 'three' in road_type or '3' in road_type:
                return 0  # three-way junction
            elif 'four' in road_type or '4' in road_type:
                return 1  # four-way junction
            else:
                return 1  # 默认为四向路口
        elif 'straight' in road_type:
            return 2
        elif 'curve' in road_type or 'bend' in road_type:
            return 3

        return 4  # unknown

    def _augment_trajectory(self, trajectory):
        device = trajectory.device
        # 转换为Tensor处理前确保输入类型
        if isinstance(trajectory, np.ndarray):
            trajectory = torch.FloatTensor(trajectory)

        if torch.rand(1, device=device) < self.augment_prob:
            angle = torch.empty(1, device=device).uniform_(-45, 45)
            rad = torch.deg2rad(angle)
            rot_mat = torch.tensor([
                [torch.cos(rad), -torch.sin(rad)],
                [torch.sin(rad), torch.cos(rad)]
            ], device=device)
            trajectory[..., :2] = torch.matmul(trajectory[..., :2], rot_mat.T)

        if torch.rand(1, device=device) < self.augment_prob:
            noise = torch.randn_like(trajectory) * self.noise_scale.to(device)
            trajectory += noise * torch.tensor([1.0, 1.0, 0.5, 2.0, 2.0], device=device)

        if torch.rand(1) < 0.3:
            mask_len = int(trajectory.size(0) * 0.2)
            start = torch.randint(0, trajectory.size(0) - mask_len, (1,))
            trajectory[start:start + mask_len] = 0

        # 新增地图感知增强
        if torch.rand(1) < 0.5:
            # 沿道路方向平移增强
            dx = torch.randn(1) * 2.0  # 最大2米偏移
            dy = torch.randn(1) * 0.5
            trajectory[..., 0] += dx
            trajectory[..., 1] += dy

        if hasattr(self, 'current_map_meta') and torch.rand(1).item() < 0.3:
            # 修改曲率计算方式
            curv = self.current_map_meta[:, 1].mean().item()  # 转换为Python float
            rotate_angle = curv * 10
            # 修正clamp参数类型
            rad = torch.deg2rad(
                torch.clamp(
                    torch.tensor(rotate_angle, device=device),
                    min=-15,
                    max=15
                )
            )
            rot_mat = torch.tensor([
                [torch.cos(rad), -torch.sin(rad)],
                [torch.sin(rad), torch.cos(rad)]
            ], device=device)
            trajectory[..., :2] = torch.matmul(trajectory[..., :2], rot_mat.T)

        new_noise_scale = torch.tensor([0.1, 0.1, 0.05, 0.2, 0.2], device=trajectory.device)
        if torch.rand(1) < self.augment_prob:
            noise = torch.randn_like(trajectory) * new_noise_scale
            trajectory += noise * torch.tensor([1.0, 1.0, 0.3, 1.5, 1.5], device=device)

        return trajectory

# -------------------- 模型组件 --------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels))
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv(x) + self.shortcut(x))


class EnhancedTrajectoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(5, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(128),
            ResidualBlock(128, 256, kernel_size=5),
            ResidualBlock(256, 256, kernel_size=3),
            nn.AdaptiveMaxPool1d(20)
        )
        self.attention = nn.MultiheadAttention(256, 8, dropout=0.2)
        self.gru = nn.GRU(256, 512, bidirectional=True, batch_first=True)
        self.feature_enhancer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3))

    def forward(self, x):
        x = self.conv_block(x.transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        gru_out, _ = self.gru(x)
        return self.feature_enhancer(gru_out[:, -1])


class MultiAgentInteraction(nn.Module):
    def __init__(self, agent_dim=128, map_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        # 新增地图特征投影层
        self.map_proj = nn.Linear(map_dim, agent_dim)

        # 轨迹编码器 (LSTM)
        self.traj_encoder = nn.LSTM(
            input_size=5,
            hidden_size=agent_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.traj_proj = nn.Linear(agent_dim * 2, agent_dim)

        # Agent-Agent 交互
        self.agent_self_attn = nn.MultiheadAttention(
            agent_dim, num_heads, dropout=dropout
        )

        # Agent-Map 交互
        self.agent_map_attn = nn.MultiheadAttention(
            agent_dim, num_heads, dropout=dropout
        )

        # 相对位置编码器
        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, agent_dim)
        )

        self.num_heads = num_heads

    def forward(self, agent_trajs, map_emb, agent_ids):
        """
        Args:
            agent_trajs: [B, A, T, 5]  (A: 实际智能体数量)
            map_emb: [B, P, D_map]      地图嵌入 (P: 多边形数量)
            agent_ids: [B, A]            智能体ID
        """
        B, A, T, _ = agent_trajs.shape
        P = map_emb.size(1)

        # 1. 轨迹编码 (LSTM)
        agent_feats = []
        for a in range(A):
            traj = agent_trajs[:, a]  # [B, T, 5]
            output, _ = self.traj_encoder(traj)
            agent_feat = output[:, -1]  # [B, 2*agent_dim]
            agent_feat = self.traj_proj(agent_feat)  # [B, agent_dim]
            agent_feats.append(agent_feat)
        agent_feats = torch.stack(agent_feats, dim=1)  # [B, A, agent_dim]

        # 2. Agent-Agent 交互
        # 计算相对位置编码
        rel_pos_feats = self._calc_relative_positions(agent_trajs, agent_ids)
        rel_pos_emb = self.rel_pos_encoder(rel_pos_feats)  # [B, A, A, agent_dim]

        valid_mask = self._create_agent_mask(agent_ids)  # [B, A, A]

        # 修正重塑逻辑
        attn_mask = valid_mask.unsqueeze(1)  # [B, 1, A, A]
        attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)  # [B, num_heads, A, A]
        attn_mask = attn_mask.reshape(B * self.num_heads, A, A)  # [B*num_heads, A, A]

        # 应用多头自注意力
        agent_feats_trans = agent_feats.transpose(0, 1)  # [A, B, agent_dim]
        agent_feats_attn, attn_weights = self.agent_self_attn(
            agent_feats_trans,
            agent_feats_trans,
            agent_feats_trans,
            attn_mask=~attn_mask  # 注意：PyTorch中True表示需要屏蔽
        )
        agent_feats_attn = agent_feats_attn.transpose(0, 1)  # [B, A, agent_dim]

        # 添加相对位置编码
        agent_feats = agent_feats + agent_feats_attn + rel_pos_emb.mean(dim=2)

        # 3. Agent-Map 交互
        # 直接使用三维地图特征 [B, P, D_map]
        map_feat = self.map_proj(map_emb)  # [B, P, agent_dim]

        # 重塑为注意力需要的格式: [P, B, agent_dim]
        map_feat = map_feat.permute(1, 0, 2)  # [P, B, agent_dim]

        agent_feats_trans = agent_feats.permute(1, 0, 2)  # [A, B, agent_dim]

        agent_feats_map, _ = self.agent_map_attn(
            agent_feats_trans,  # [A, B, agent_dim]
            map_feat,  # [P, B, agent_dim]
            map_feat  # [P, B, agent_dim]
        )
        agent_feats_map = agent_feats_map.permute(1, 0, 2)  # [B, A, agent_dim]
        agent_feats = agent_feats + agent_feats_map

        # 提取交互矩阵用于可视化
        interaction_matrix = self._create_interaction_matrix(
            agent_feats, attn_weights, rel_pos_emb
        )

        return agent_feats, interaction_matrix

    def _calc_relative_positions(self, agent_trajs, agent_ids):
        """计算agent间的相对位置和朝向"""
        B, A, T, _ = agent_trajs.shape
        # 使用最后位置
        pos = agent_trajs[:, :, -1, :2]  # [B, A, 2]
        yaw = agent_trajs[:, :, -1, 2]  # [B, A]

        # 扩展以计算成对关系
        pos1 = pos.unsqueeze(2)  # [B, A, 1, 2]
        pos2 = pos.unsqueeze(1)  # [B, 1, A, 2]
        yaw1 = yaw.unsqueeze(2)  # [B, A, 1]
        yaw2 = yaw.unsqueeze(1)  # [B, 1, A]

        # 相对位移
        dx = pos2[..., 0] - pos1[..., 0]  # [B, A, A]
        dy = pos2[..., 1] - pos1[..., 1]  # [B, A, A]

        # 相对朝向
        yaw_diff = yaw2 - yaw1  # [B, A, A]
        sin_diff = torch.sin(yaw_diff)
        cos_diff = torch.cos(yaw_diff)

        rel_pos = torch.stack([dx, dy, sin_diff, cos_diff], dim=-1)  # [B, A, A, 4]

        # 掩码自身交互
        self_mask = torch.eye(A, dtype=torch.bool, device=agent_trajs.device)
        rel_pos[:, self_mask] = 0

        return rel_pos

    def _create_agent_mask(self, agent_ids):
        """创建有效agent的掩码"""
        # agent_id = -1 表示无效agent
        valid_mask = (agent_ids != -1).unsqueeze(1) & (agent_ids != -1).unsqueeze(2)
        return valid_mask

    def _create_interaction_matrix(self, agent_feats, attn_weights, rel_pos_emb):
        """创建用于可视化的交互矩阵"""
        B, A, D = agent_feats.shape

        # 使用特征差值而不是外积
        interaction = agent_feats.unsqueeze(2) - agent_feats.unsqueeze(1)  # [B, A, A, D]

        # 处理注意力权重 - 更健壮的处理
        if attn_weights is not None:
            # 获取实际注意力权重形状
            orig_shape = attn_weights.shape
            num_heads = self.num_heads

            # 检查是否可以重塑为 [B, num_heads, A, A]
            if orig_shape.numel() == B * num_heads * A * A:
                attn_weights = attn_weights.view(B, num_heads, A, A)
                attn_weights_mean = attn_weights.mean(dim=1)  # [B, A, A]
                attn_weights_mean = attn_weights_mean.unsqueeze(-1)  # [B, A, A, 1]
                interaction = interaction * attn_weights_mean
            else:
                # 记录实际形状用于调试
                logger.debug(
                    f"注意力权重形状不匹配: 预期{B * num_heads * A * A}元素, "
                    f"实际{orig_shape.numel()}元素. 形状: {orig_shape}"
                )
        return interaction

# -------------------- 增强型地图编码器 --------------------
class EnhancedVectorNetEncoder(nn.Module):
    def __init__(self, poly_dim=6, meta_dim=4, hidden_dim=128):
        super().__init__()
        self.poly_encoder = nn.Sequential(
            nn.Linear(poly_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        self.global_interaction = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim + hidden_dim // 2,  # 修正输入维度
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=3
        )
        self.type_emb = nn.Embedding(5, hidden_dim)
        # 调整投影层输入维度
        self.proj = nn.Linear(hidden_dim + hidden_dim // 2 + 2, 512)

    def forward(self, map_data):
        B, P, N, _ = map_data['polylines'].shape

        # 处理道路多边形特征
        poly_input = map_data['polylines'].view(B * P * N, -1)  # [总点数, 6]
        poly_feats = self.poly_encoder(poly_input)  # [B*P*N, hidden_dim]

        # 加入道路类型嵌入
        road_types = map_data['polylines'][..., 4].long().view(-1)  # [B*P*N]
        poly_feats += self.type_emb(road_types)  # [B*P*N, hidden_dim]

        # 按多边形聚合特征
        poly_feats = poly_feats.view(B * P, N, -1).mean(dim=1)  # [B*P, hidden_dim]

        # 处理元数据特征
        meta_input = map_data['poly_meta'].view(B * P, -1)  # [B*P, 4]
        meta_feats = self.meta_encoder(meta_input)  # [B*P, hidden_dim//2]

        # 特征融合与全局交互
        fused = torch.cat([poly_feats, meta_feats], dim=1)  # [B*P, hidden_dim + hidden_dim//2]
        global_feats = self.global_interaction(fused.view(P, B, -1))  # [P, B, D]

        # 扩展全局特征
        global_mean = global_feats.mean(dim=0)  # [B, D]
        global_expanded = global_mean.repeat_interleave(P, dim=0)  # [B*P, D]

        # 空间特征提取
        spatial_feats = torch.cat([
            map_data['poly_meta'][..., 0].unsqueeze(-1),  # 道路长度
            map_data['poly_meta'][..., 1].unsqueeze(-1)  # 曲率
        ], dim=-1).view(B * P, -1)  # [B*P, 2]

        # 最终特征融合
        combined = torch.cat([global_expanded, spatial_feats], dim=1)  # [B*P, D+2]
        output = self.proj(combined)  # [B*P, 512]
        return output.view(B, P, -1)  # 修正为 [B, P, 512]


# -------------------- 改进的LLM接口模块（DeepSeek版本） --------------------
class DeepSeekDecisionProcessor:
    def __init__(self, cache_dir=os.path.expanduser("~/llm_cache")):
        self.client = OpenAI(
            api_key="sk-34dce718d35942c78f0facfe15c83029",
            base_url="https://api.deepseek.com/v1",
            timeout=30.0  # 根据V3模型优化缩短超时:cite[2]
        )
        self.cache_file = os.path.join(cache_dir, "llm_cache_v3.json")
        self.retry_limit = 3
        self._cache_hits = 0
        self.logger = logging.getLogger(__name__)
        self.cache = self._load_cache()

        # 基于CODEI/O方法的参数校验标准:cite[1]
        self.validation_rules = {
            "yaw_range": lambda x: x[0] < x[1],
            "speed_range": lambda x: x[0] >= 0 and x[1] > x[0],
            "risk_level": lambda x: x in ["LOW", "MEDIUM", "HIGH"]
        }

    def _load_cache(self) -> Dict:
        """加载本地缓存文件"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"缓存加载失败: {str(e)}")
        return {}

    def save_cache(self) -> None:
        """增强鲁棒性的缓存保存"""
        try:
            safe_cache = {}
            for k, v in self.cache.items():
                if isinstance(v, dict) and 'path_decision' in v and 'speed_decision' in v:
                    safe_cache[k] = v
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(safe_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"缓存保存失败: {str(e)}")

    def _generate_cache_key(self, features: Dict) -> str:
        """更稳定的特征哈希（包含numpy类型转换）"""

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):  # 处理numpy标量类型
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        stable_features = {
            'traj': features.get('traj', [])[:10],  # 取前10帧关键特征
            'map': features.get('map_data', {}).get('curvature', 0)
        }
        # 深度转换所有numpy类型
        stable_features = convert_numpy(stable_features)

        return hashlib.sha256(
            json.dumps(stable_features, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()

    def _preprocess_features(self, features: Dict) -> Dict:
        """统一输入特征格式（关键新增方法）"""
        # 处理张量输入
        if isinstance(features, torch.Tensor):
            processed = features.detach().cpu().numpy().tolist()
        # 处理numpy数组
        elif isinstance(features, np.ndarray):
            processed = features.tolist()
        # 其他类型直接传递
        else:
            processed = features

        # 确保特征格式统一
        if not isinstance(processed, dict) or 'traj' not in processed:
            processed = {'traj': processed}

        return processed

    def process(self, traj_features: Dict) -> Dict:
        """处理轨迹特征并生成决策"""
        processed_features = self._preprocess_features(traj_features)
        cache_key = self._generate_cache_key(processed_features)

        if cache_key in self.cache:
            self._cache_hits += 1
            return self._format_cached_result(self.cache[cache_key])

        for attempt in range(self.retry_limit):
            try:
                result = self._call_api_with_validation(processed_features)
                self.cache[cache_key] = result
                self.save_cache()
                return result
            except (APITimeoutError, APIConnectionError, ValueError) as e:  # 修改为已导入的异常类型
                self.logger.warning(f"API尝试{attempt + 1}次失败: {str(e)}")
                time.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"严重错误: {str(e)}")
                break

        return self._get_default_decision()

    def _call_api_with_validation(self, features: Dict) -> Dict:
        """调用API并验证响应（含自动修正）"""
        try:
            # 添加详细的调试日志
            self.logger.debug("正在准备API请求数据...")

            prompt = self._build_prompt(features)
            start_time = time.time()

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=200
            )

            # 添加性能监控
            latency = time.time() - start_time
            self.logger.info(f"API调用成功 耗时:{latency:.2f}s Token用量:{response.usage.total_tokens}")

            raw_response = response.choices[0].message.content
            return self._process_raw_response(raw_response)

        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            raise

    def _build_prompt(self, features: Dict) -> str:
        """增强的提示词构建，包含更多上下文信息"""
        # 提取历史轨迹特征
        traj = features.get('traj', [])
        if len(traj) == 0:
            return self._get_default_decision()

        # 轨迹统计特征
        speeds = [np.linalg.norm(point[3:5]) for point in traj]
        avg_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0

        # 计算当前速度（轨迹最后一帧的速度）
        current_speed = speeds[-1] if speeds else 0.0
        delta_speed = current_speed - avg_speed

        # 计算航向变化
        yaw_changes = np.diff([p[2] for p in traj])
        avg_yaw_change = np.degrees(np.mean(yaw_changes)) if len(yaw_changes) > 0 else 0

        # 地图特征
        curvature = features.get('map_data', {}).get('curvature', 0)
        lane_type = features.get('map_data', {}).get('lane_type', 0)
        lane_types = ["普通车道", "公交专用道", "应急车道", "匝道", "十字路口"][lane_type]

        # 车辆动力学参数
        dynamics = features.get('dynamics', {})

        prompt = f"""自动驾驶决策生成任务（版本4.0）：
                 请遵循以下推理步骤：
        1. **状态分析**:
           - 当前速度: {current_speed:.1f}m/s (历史平均: {avg_speed:.1f}m/s)
           - 航向变化: {avg_yaw_change:.1f}°/s
           - 道路曲率: {curvature:.2f} rad/m
           - 车道类型: {lane_types}

        2. **风险评估**:
           - 横向风险: {"高" if abs(avg_yaw_change) > 15 else "中" if abs(avg_yaw_change) > 5 else "低"}
           - 纵向风险: {"高" if abs(delta_speed) > 3 else "中" if abs(delta_speed) > 1 else "低"}
           - 环境风险: {"高" if lane_type in [2, 3] else "中" if lane_type == 4 else "低"} (应急车道/匝道/路口)

        3. **约束分析**:
           - 最大转向角: {np.degrees(dynamics.get('max_steer', 0.6)):.1f}°
           - 加速能力: {dynamics.get('max_accel', 3)}m/s² 
           - 制动能力: {dynamics.get('max_decel', 5)}m/s²
           - 横向加速度限值: {dynamics.get('max_lat_acc', 3.0)}m/s²

        4. **决策生成**:
           - 基于以上分析，选择最安全的决策组合
           - 考虑: 风险平衡 + 效率优化 + 舒适性

        5. **参数设定**:
           - 根据决策类型设定运动学参数边界
           - 确保参数在物理可行范围内

        候选决策集：
        [路径] FOLLOW(保持) | LEFT_CHANGE(左变) | RIGHT_CHANGE(右变)
        [速度] KEEP(保持) | ACCEL(加速) | DECEL(减速) | STOP(急停)

        输出要求（JSON格式）：
                {{
                    "path_decision": "路径决策",
                    "speed_decision": "速度决策",
                    "confidence": 决策置信度(0.1~1.0),
                    "constraints": {{
                        "max_lateral_acc": 最大横向加速度(m/s²),
                        "target_speed": 目标速度(m/s),
                        "steer_change_rate": 转向角变化率(°/s)
                    }},
                    "rationale": "决策依据（结合场景特征分析）"
                }}"""

        return prompt

    def _get_road_curvature(self, features):
        """从地图数据获取道路曲率"""
        if 'map_data' in features:
            return np.mean(features['map_data']['poly_meta'][:, 1])
        return 0.0

    def _validate_result(self, result: Dict, silent=False) -> bool:
        """验证决策参数有效性"""
        error_msgs = []
        for field, validator in self.validation_rules.items():
            if not validator(result.get(field, None)):
                error_msgs.append(f"{field}验证失败")

        if error_msgs and not silent:
            raise ValueError(" | ".join(error_msgs))
        return len(error_msgs) == 0

    def _format_cached_result(self, cached_data: Dict) -> Dict:
        """格式化缓存结果（含单位转换）"""
        return {
            "path_decision": cached_data.get("path_decision", "FOLLOW"),
            "speed_decision": cached_data.get("speed_decision", "KEEP"),
        }

    def _get_default_decision(self) -> Dict:
        """智能默认决策生成"""
        return {
            "path_decision": "FOLLOW",
            "speed_decision": "KEEP",
            "rationale": "默认安全决策"
        }

    def decision_to_mask(self, decision: Dict) -> Dict[str, torch.Tensor]:
        """将决策转换为模型可用的张量掩码"""
        return {
            "yaw_range": torch.tensor(decision["yaw_range"]),
            "speed_range": torch.tensor(decision["speed_range"]),
            "risk_level": torch.tensor(
                {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[decision["risk_level"]]
            )
        }

    def _process_raw_response(self, raw_response: str) -> Dict:
        """增强的响应处理与验证"""
        try:
            result = json.loads(raw_response)
            # 关键验证字段
            if "path_decision" not in result or "speed_decision" not in result:
                raise ValueError("响应缺少必要字段")

            # 决策有效性检查
            valid_paths = ["FOLLOW", "LEFT_CHANGE", "RIGHT_CHANGE"]
            # 修复：支持"DECEL"和"ACCEL"缩写
            valid_speeds = ["KEEP", "ACCEL", "ACCELERATE", "DECEL", "DECELERATE", "STOP"]

            # 统一转换为大写
            path_decision = result["path_decision"].upper()
            speed_decision = result["speed_decision"].upper()

            # 映射缩写到完整形式
            if speed_decision == "ACCEL":
                speed_decision = "ACCELERATE"
            elif speed_decision == "DECEL":
                speed_decision = "DECELERATE"

            result["path_decision"] = path_decision
            result["speed_decision"] = speed_decision

            if path_decision not in valid_paths:
                raise ValueError(f"无效路径决策: {path_decision}")
            if speed_decision not in valid_speeds:
                raise ValueError(f"无效速度决策: {speed_decision}")

            return result
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            json_str = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_str:
                return self._process_raw_response(json_str.group())
            return self._get_default_decision()
        except Exception as e:
            logger.warning(f"响应解析失败: {str(e)}，使用默认决策")
            return self._get_default_decision()

# -------------------- 解释器 --------------------
class DecisionExplainer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 决策生成组件
        self.path_decision = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3种路径决策: 左转/保持/右转
        )

        self.speed_decision = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3种速度决策: 加速/保持/减速
        )

        # 解释生成器 (小型Transformer)
        self.explainer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=512
            ),
            num_layers=2
        )

        # 文本解码器
        self.text_decoder = nn.Linear(hidden_dim, 300)  # 300维词向量空间

    def forward(self, decision_data):
        """
        生成决策解释:
        {
            "salient_features": [特征描述列表],
            "potential_actions": [潜在行为描述],
            "final_decision": {
                "intent": "数字, 决策描述",
                "trajectory": "轨迹描述",
                "explanation": "解释文本"
            }
        }
        """
        # 提取关键特征
        ego_feat = decision_data['agent_feats'][:, 0]  # ego特征 [B, 128]
        context_feat = self.feature_extractor(ego_feat)  # [B, 256]

        # 生成决策
        path_logits = self.path_decision(context_feat)
        speed_logits = self.speed_decision(context_feat)

        path_decision = torch.argmax(path_logits, dim=1)
        speed_decision = torch.argmax(speed_logits, dim=1)

        # 生成决策代码
        decision_code = path_decision * 3 + speed_decision  # 0-8的决策代码

        # 只处理批次中的第一个样本
        first_decision_code = decision_code[0] if decision_code.numel() > 0 else 4  # 默认为保持保持

        explanation = self._generate_explanation(
            context_feat[0], first_decision_code, decision_data
        )

        # 生成显著特征描述
        salient_features = self._extract_salient_features(decision_data)

        return {
            "salient_features": salient_features,
            "potential_actions": self._generate_potential_actions(first_decision_code),
            "final_decision": {
                "intent": f"{first_decision_code.item()}, {self._decode_decision(first_decision_code)}",
                "trajectory": explanation["trajectory"],
                "explanation": explanation["text"]
            }
        }

    def _generate_explanation(self, context_feat, decision_code, decision_data):
        # 使用Transformer生成解释
        memory = self.explainer(context_feat.unsqueeze(0))
        text_emb = self.text_decoder(memory.squeeze(0))

        # 基于决策代码生成解释文本
        decision_text = self._decode_decision(decision_code)

        # 获取实际场景特征
        current_speed = torch.norm(decision_data['traj_feat']['input_traj'][0, -1, 3:5]).item()
        min_distance = self._get_min_distance(decision_data)
        road_type = self._get_road_type(decision_data)
        interaction_strength = self._get_interaction_strength(decision_data)

        # 生成解释文本
        explanation_text = (
            f"基于当前交通状况，我选择{decision_text}。"
            f"当前车速{current_speed:.1f}m/s，"
            f"最近车辆距离{min_distance:.1f}m，"
            f"道路类型为{road_type}。"
            f"{interaction_strength}因此这是最安全的决策。"
        )

        # 轨迹描述
        trajectory_desc = self._get_trajectory_description(decision_code)

        return {
            "text": explanation_text,
            "trajectory": trajectory_desc
        }

    def _decode_decision(self, decision_code):
        """将决策代码转换为文本描述"""
        # 确保决策代码是整数类型
        if isinstance(decision_code, torch.Tensor):
            decision_code = decision_code.item()  # 将张量转换为Python整数

        path_actions = ["左转", "保持", "右转"]
        speed_actions = ["加速", "保持", "减速"]

        path_idx = decision_code // 3
        speed_idx = decision_code % 3

        return f"{speed_actions[speed_idx]}{path_actions[path_idx]}"

    def _generate_explanation(self, context_feat, decision_code, decision_data):
        # 使用Transformer生成解释
        memory = self.explainer(context_feat.unsqueeze(0))
        text_emb = self.text_decoder(memory.squeeze(0))

        # 基于决策代码生成解释文本
        decision_text = self._decode_decision(decision_code)

        # 获取实际场景特征
        current_speed = torch.norm(decision_data['traj_feat']['input_traj'][0, -1, 3:5]).item()
        min_distance = self._get_min_distance(decision_data)
        road_type = self._get_road_type(decision_data)
        interaction_strength = self._get_interaction_strength(decision_data)

        # 生成解释文本
        explanation_text = (
            f"基于当前交通状况，我选择{decision_text}。"
            f"当前车速{current_speed:.1f}m/s，"
            f"最近车辆距离{min_distance:.1f}m，"
            f"道路类型为{road_type}。"
            f"{interaction_strength}因此这是最安全的决策。"
        )

        # 轨迹描述
        trajectory_desc = self._get_trajectory_description(decision_code)

        return {
            "text": explanation_text,
            "trajectory": trajectory_desc
        }


    def _extract_salient_features(self, decision_data):
        """提取用于决策的显著特征"""
        features = []

        # 1. Ego车辆状态
        ego_traj = decision_data['traj_feat']['input_traj'][0, -1]
        features.append(f"Ego速度: {torch.norm(ego_traj[3:5]).item():.1f}m/s")

        # 2. 最近车辆距离
        min_distance = self._get_min_distance(decision_data)
        features.append(f"最近车辆距离: {min_distance:.1f}m")

        # 3. 道路特征
        road_type = self._get_road_type(decision_data)
        features.append(f"道路类型: {road_type}")

        # 4. 交互强度
        interaction_desc = self._get_interaction_strength(decision_data)
        features.append(interaction_desc)

        return features

    def _get_min_distance(self, decision_data):
        """计算到最近车辆的实际距离"""
        ego_pos = decision_data['traj_feat']['input_traj'][0, -1, :2]
        min_distance = float('inf')
        device = ego_pos.device  # 获取当前设备

        # 检查其他车辆
        for other_agent in decision_data['traj_feat']['other_trajs']:
            if len(other_agent) > 0:
                # 确保张量在相同设备
                other_traj = other_agent[0]['traj']
                if isinstance(other_traj, np.ndarray):
                    other_pos = torch.tensor(
                        other_traj[-1, :2],
                        device=device,
                        dtype=torch.float32
                    )
                elif isinstance(other_traj, torch.Tensor):
                    other_pos = other_traj[-1, :2].to(device)
                else:
                    continue

                distance = torch.norm(ego_pos - other_pos).item()
                if distance < min_distance:
                    min_distance = distance

        return min_distance if min_distance < float('inf') else 0.0

    def _get_road_type(self, decision_data):
        """获取实际道路类型"""
        # 直接从决策数据中获取道路类型
        return decision_data['traj_feat']['road_type'][0].item()

    def _decode_road_type(self, road_type_id):
        types = {
            0: "三岔路口",
            1: "四岔路口",
            2: "直道",
            3: "弯道",
            4: "未知"
        }
        return types.get(int(road_type_id), "未知")

    def _get_interaction_strength(self, decision_data):
        """分析实际交互强度"""
        interaction_matrix = decision_data['interaction_matrix'][0, 0]
        avg_interaction = interaction_matrix.mean().item()

        if avg_interaction > 0.7:
            return "与周围车辆有强交互"
        elif avg_interaction > 0.4:
            return "与周围车辆有中等交互"
        else:
            return "与周围车辆交互较弱"

    def _get_trajectory_description(self, decision_code):
        """生成轨迹描述"""
        path_idx = decision_code // 3
        speed_idx = decision_code % 3

        if path_idx == 0:  # 左转
            if speed_idx == 0:  # 加速
                return "轨迹将快速向左变道，同时保持安全距离"
            elif speed_idx == 1:  # 保持
                return "轨迹将平稳向左变道，保持当前速度"
            else:  # 减速
                return "轨迹将谨慎向左变道，适当减速"

        elif path_idx == 1:  # 保持
            if speed_idx == 0:  # 加速
                return "轨迹将沿当前车道加速行驶"
            elif speed_idx == 1:  # 保持
                return "轨迹将沿当前车道保持速度行驶"
            else:  # 减速
                return "轨迹将沿当前车道减速行驶"

        else:  # 右转
            if speed_idx == 0:  # 加速
                return "轨迹将快速向右变道，同时保持安全距离"
            elif speed_idx == 1:  # 保持
                return "轨迹将平稳向右变道，保持当前速度"
            else:  # 减速
                return "轨迹将谨慎向右变道，适当减速"

    def _generate_potential_actions(self, decision_code):
        """生成潜在行为描述"""
        actions = []

        # 当前决策
        current = self._decode_decision(decision_code)
        actions.append(f"当前选择: {current}")

        # 备选决策
        if decision_code > 0:
            alt = self._decode_decision(decision_code - 1)
            actions.append(f"备选方案: {alt} (风险稍高)")

        if decision_code < 8:
            alt = self._decode_decision(decision_code + 1)
            actions.append(f"备选方案: {alt} (效率稍低)")

        # 添加安全决策
        safe_code = 4  # 保持保持
        if decision_code != safe_code:
            safe_action = self._decode_decision(safe_code)
            actions.append(f"安全方案: {safe_action} (最保守)")

        return actions


# -------------------- 约束生成器 --------------------
class ConstraintGenerator:
    def __init__(self, vehicle_params, map_info):
        self.vehicle = vehicle_params
        self.map_info = map_info

        # 决策-约束映射表（包含12种典型决策组合）
        self.decision_constraint_map = {
            # (路径决策, 速度决策): 约束参数
            ('FOLLOW', 'KEEP'): {
                'max_lat_acc': 1.0,  # 最大横向加速度 (m/s²)
                'speed_ratio': 1.0,  # 目标速度系数
                'steer_rate': 10.0  # 转向角变化率 (°/s)
            },
            ('FOLLOW', 'ACCELERATE'): {
                'max_lat_acc': 0.8,
                'speed_ratio': 1.2,
                'steer_rate': 8.0
            },
            ('FOLLOW', 'DECELERATE'): {
                'max_lat_acc': 0.6,
                'speed_ratio': 0.7,
                'steer_rate': 5.0
            },
            ('LEFT_CHANGE', 'KEEP'): {
                'max_lat_acc': 1.8,
                'speed_ratio': 0.9,
                'steer_rate': 15.0
            },
            ('LEFT_CHANGE', 'ACCELERATE'): {
                'max_lat_acc': 1.5,
                'speed_ratio': 1.1,
                'steer_rate': 12.0
            },
            ('LEFT_CHANGE', 'DECELERATE'): {
                'max_lat_acc': 1.2,
                'speed_ratio': 0.8,
                'steer_rate': 10.0
            },
            ('RIGHT_CHANGE', 'KEEP'): {
                'max_lat_acc': 1.8,
                'speed_ratio': 0.9,
                'steer_rate': 15.0
            },
            ('RIGHT_CHANGE', 'ACCELERATE'): {
                'max_lat_acc': 1.5,
                'speed_ratio': 1.1,
                'steer_rate': 12.0
            },
            ('RIGHT_CHANGE', 'DECELERATE'): {
                'max_lat_acc': 1.2,
                'speed_ratio': 0.8,
                'steer_rate': 10.0
            },
            # 特殊场景处理
            ('FOLLOW', 'STOP'): {
                'max_lat_acc': 0.3,
                'speed_ratio': 0.0,
                'steer_rate': 5.0
            },
            ('LEFT_CHANGE', 'STOP'): {
                'max_lat_acc': 0.5,
                'speed_ratio': 0.3,
                'steer_rate': 8.0
            },
            ('RIGHT_CHANGE', 'STOP'): {
                'max_lat_acc': 0.5,
                'speed_ratio': 0.3,
                'steer_rate': 8.0
            }
        }

        # 道路类型影响系数 [普通, 公交道, 应急道, 匝道, 交叉口]
        self.road_type_factors = [1.0, 0.9, 1.2, 1.5, 0.7]

        # 曲率-速度修正曲线参数
        self.curvature_speed_factor = lambda c: 1.0 / (1.0 + 2.5 * abs(c))

    def generate(self, decision: Dict, current_state: Dict) -> Dict:
        """生成综合运动约束
        Args:
            decision: LLM决策字典，包含path_decision和speed_decision
            current_state: 当前车辆状态字典
        Returns:
            包含完整约束参数的字典
        """
        # 基础安全约束计算
        base_constraint = self._calc_base_constraint(current_state)

        # 获取决策参数（使用默认参数作为fallback）
        decision_key = (decision['path_decision'], decision['speed_decision'])
        decision_params = self.decision_constraint_map.get(
            decision_key,
            {'max_lat_acc': 1.0, 'speed_ratio': 1.0, 'steer_rate': 10.0}
        )

        # 应用道路环境修正
        road_type = int(self.map_info.get('lane_type', 0))
        road_type = max(0, min(road_type, 4))
        road_factor = self.road_type_factors[road_type]
        curvature = self.map_info.get('curvature', 0)

        # 动态调整约束参数
        adjusted_params = {
            'max_lat_acc': decision_params['max_lat_acc'] * road_factor,
            'target_speed': self._calc_target_speed(current_state, decision_params, curvature),
            'steer_rate': self._calc_steer_rate(current_state, decision_params)
        }

        # 生成最终约束集
        return {
            # 横向控制约束
            'yaw_range': self._calc_yaw_constraint(decision, current_state, base_constraint),
            'max_lat_acc': min(adjusted_params['max_lat_acc'], self.vehicle['max_lat_acc']),

            # 纵向控制约束
            'speed_range': self._calc_speed_constraint(current_state, adjusted_params),
            'max_accel': self._calc_safe_accel(current_state),

            # 执行器约束
            'max_steer_rate': adjusted_params['steer_rate'],
            'steer_ratio_range': self._calc_steer_ratio(decision, current_state),

            # 环境约束
            'curvature_factor': self.curvature_speed_factor(curvature),
            'road_type': road_type
        }

    def _calc_base_constraint(self, state):
        """计算基础安全约束"""
        current_speed = state['speed']
        return {
            'yaw_range': [
                state['yaw'] - math.radians(5),
                state['yaw'] + math.radians(5)
            ],
            'speed_range': [
                max(current_speed - 2.0, 0),
                min(current_speed + 2.0, self.vehicle['speed_limit'])
            ]
        }

    def _calc_target_speed(self, state, params, curvature):
        """计算目标速度"""
        base_speed = state['speed'] * params['speed_ratio']
        curvature_factor = self.curvature_speed_factor(curvature)
        return min(
            base_speed * curvature_factor,
            self.map_info.get('speed_limit', 20.0)
        )

    def _calc_steer_rate(self, state, params):
        """计算安全转向速率"""
        current_steer = abs(state['yaw'] - state['target_yaw'])
        remaining_steer = self.vehicle['max_steer'] - current_steer
        return min(params['steer_rate'], remaining_steer * 2.0)  # 剩余转向空间越大，允许更快转向

    def _calc_yaw_constraint(self, decision, state, base):
        """生成航向角约束"""
        if decision['path_decision'] == 'FOLLOW':
            return base['yaw_range']

        steer_direction = 1 if 'LEFT' in decision['path_decision'] else -1
        target_yaw = state['yaw'] + steer_direction * self.vehicle['max_steer'] * 0.8
        safety_margin = math.radians(3)  # 3度安全余量

        return [
            min(target_yaw - safety_margin, state['yaw']),
            max(target_yaw + safety_margin, state['yaw'])
        ]

    def _calc_speed_constraint(self, state, params):
        """生成速度约束"""
        target = params['target_speed']
        return [
            max(target - 1.0, 0),
            min(target + 1.0, self.vehicle['speed_limit'])
        ]

    def _calc_steer_ratio(self, decision, state):
        """计算转向比例约束"""
        if decision['path_decision'] == 'FOLLOW':
            return [0.7, 1.3]  # 允许±30%的转向比例波动
        else:
            return [0.9, 1.1]  # 变道时严格控制转向比例

    def _calc_safe_accel(self, state):
        """基于跟车模型的安全加速度"""
        leading_dist = state.get('leading_distance', 20.0)
        safe_dist = state['speed'] * 2.5  # 2.5秒时距
        if leading_dist < safe_dist:
            return min(
                (leading_dist - safe_dist) / 2.5,
                self.vehicle['max_decel']
            )
        return self.vehicle['max_accel']


# -------------------- 改进的层次预测模型 --------------------
class EnhancedHierarchicalPredictor(nn.Module):
    def __init__(self, num_modes=5, use_llm=True, vehicle_params=None, scalers=None):
        super().__init__()
        self.map_encoder = EnhancedVectorNetEncoder()
        # 新增多智能体交互模块
        self.agent_interaction = MultiAgentInteraction()
        self.traj_encoder = nn.Sequential(
            nn.Conv1d(5, 128, 5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(128),
            ResidualBlock(128, 256),
            nn.AdaptiveMaxPool1d(20)
        )
        self.traj_attention = nn.MultiheadAttention(256, 8)
        self.traj_gru = nn.GRU(256, 512, bidirectional=True, batch_first=True)
        self.traj_proj = nn.Linear(1024, 512)

        self.decision_processor = DeepSeekDecisionProcessor() if use_llm else None
        self.decision_encoder = nn.Sequential(
            nn.Embedding(6, 32),  # 横向3类+纵向3类
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )
        self.decision_explainer = DecisionExplainer(input_dim=128)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,  # 与轨迹嵌入维度匹配
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 约束相关组件
        self.vehicle_params = vehicle_params or {
            'max_steer': math.radians(30),
            'max_accel': 3.0,
            'max_decel': 5.0,
            'speed_limit': 16.67
        }
        self.constraint_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.intent_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 6)  # 横向3类 + 纵向3类
        )

        # 融合层调整
        self.fusion = nn.Sequential(
            nn.Linear(512 + 64 + 64, 1024),  # 正确维度640 = 512+64+64
            nn.LayerNorm(1024),
            nn.GELU()
        )
        # 新增约束特征维度参数
        self.constraint_dim = 5  # 对应yaw_low/yaw_high/speed_low/speed_high/max_lat_acc

        # 改进的解码器
        self.decoder = ConstraintAwareDecoder(
            input_dim=1024,
            hidden_dim=1024,
            pred_steps=30,
            num_modes=num_modes,
            constraint_dim=self.constraint_dim,
            vehicle_params=self.vehicle_params
        )

        self.risk_assessment = EnhancedRiskAssessment()
        self.num_modes = num_modes
        self.scalers = scalers if scalers is not None else {'position': StandardScaler(),
                                                            'velocity': StandardScaler()}  # 使用传入的scalers

        # 新增步数调度参数
        self.register_buffer('steps_schedule',
                             torch.linspace(5, 30, steps=20, dtype=torch.int))  # 20个epoch从5步到30步

    def forward(self, vector_map, traj_feat, constraints=None, teacher_forcing=None):
        # 获取批量大小
        input_traj = traj_feat['input_traj']
        B = input_traj.size(0)  # 批量大小
        T = input_traj.size(1)  # 轨迹长度
        self.road_type = traj_feat['road_type']  # 道路类型

        # ====== 关键修改：组织所有车辆轨迹数据 ======
        agent_trajs = self._gather_all_agent_trajs(traj_feat)  # [B, A, T, 5]
        agent_ids = self._get_agent_ids(traj_feat, B)  # [B, 4]

        # 地图编码
        map_output = self.map_encoder(vector_map)  # [B, P, 512]
        P = vector_map['polylines'].size(1)

        # ====== 关键修改：多智能体交互 ======
        agent_feats, interaction_matrix = self.agent_interaction(
            agent_trajs, map_output, agent_ids
        )

        # 提取ego车辆特征
        ego_feat = agent_feats[:, 0]  # [B, D_agent]

        # ====== 新增：生成决策解释 ======
        decision_data = {
            'agent_feats': agent_feats,  # [B, A, 128]
            'map_emb': map_output,  # [B, P, 512]
            'interaction_matrix': interaction_matrix,
            'traj_feat': {
                'input_traj': agent_trajs[:, 0],  # ego车辆轨迹
                'road_type': self.road_type,  # 添加道路类型
                'other_trajs': traj_feat['other_trajs']  # 其他车辆轨迹
            }
        }
        decision_rationale = self.decision_explainer(decision_data)

        # 轨迹编码（使用ego车辆的历史轨迹）
        x = self.traj_encoder(traj_feat['input_traj'].transpose(1, 2))  # [B,256,20]
        x = x.permute(2, 0, 1)  # [20,B,256]
        attn_out, _ = self.traj_attention(x, x, x)
        x = x + attn_out
        gru_out, _ = self.traj_gru(x.permute(1, 0, 2))  # [B,20,1024]
        traj_emb = self.traj_proj(gru_out[:, -1])  # [B,512]

        # 决策与约束生成（使用交互后的ego特征）
        decisions = []
        lateral_labels = []
        longitudinal_labels = []
        for i in range(B):
            features = {
                'traj': self._get_denorm_traj(traj_feat['input_traj'][i]),
                'map_data': self._extract_map_features(vector_map, i),
                'dynamics': self.vehicle_params,
                # 添加交互特征到LLM输入
                'agent_interaction': agent_feats[i].detach().cpu().numpy().tolist()
            }
            decision = self.decision_processor.process(features)
            decisions.append(decision)

            # 转换决策到类别标签
            lateral_map = {"LEFT_CHANGE": 0, "FOLLOW": 1, "RIGHT_CHANGE": 2}
            longi_map = {"DECELERATE": 0, "KEEP": 1, "ACCELERATE": 2}

            lateral_label = min(max(lateral_map.get(decision['path_decision'], 1), 0), 2)
            longi_label = min(max(longi_map.get(decision['speed_decision'], 1), 0), 2)

            lateral_labels.append(lateral_label)
            longitudinal_labels.append(longi_label)

        # 生成决策嵌入
        lateral_labels = torch.tensor(lateral_labels, device=self.device)
        longitudinal_labels = torch.tensor(longitudinal_labels, device=self.device)
        decision_feat = self.decision_encoder(
            lateral_labels * 3 + longitudinal_labels
        )  # [B,64]

        # 生成运动约束
        constraints = []
        for i, decision in enumerate(decisions):
            map_info = self._get_map_info(vector_map, i)
            current_state = self._get_current_state(traj_feat['input_traj'][i])
            constraint_gen = ConstraintGenerator(
                self.vehicle_params,
                map_info
            )
            constraints.append(constraint_gen.generate(decision, current_state))

        constraint_tensor = self._encode_constraints(constraints)  # [B,5]
        constraint_tensor = constraint_tensor.unsqueeze(1).expand(-1, self.num_modes, -1).reshape(-1, 5)  # [B*M,5]
        constraint_feat = self.constraint_encoder(constraint_tensor)  # [B*M,64]

        # 扩展地图和轨迹特征以匹配多模态
        map_emb_expanded = map_output.mean(dim=1).unsqueeze(1).expand(-1, self.num_modes, -1).reshape(-1, 512)  # [B*M,512]
        traj_emb_expanded = traj_emb.unsqueeze(1).expand(-1, self.num_modes, -1).reshape(-1, 512)  # [B*M,512]

        fused_emb, _ = self.cross_attn(
            traj_emb_expanded,  # query
            map_emb_expanded,  # key
            map_emb_expanded  # value
        )

        fused = self.fusion(torch.cat([
            fused_emb,  # [B*M, 512]
            constraint_feat,  # [B*M, 64]
            decision_feat.unsqueeze(1).expand(-1, self.num_modes, -1).reshape(-1, 64)  # [B*M, 64]
        ], dim=1))  # -> [B*M, 640]

        # 初始化解码器状态
        h_t = torch.zeros(B * self.num_modes, self.decoder.hidden_dim, device=self.device)
        c_t = torch.zeros_like(h_t)

        # 解码预测
        preds = self.decoder(
            init_state=(h_t, c_t),
            memory=fused.unsqueeze(1),
            constraints=constraint_tensor,
            teacher_forcing=teacher_forcing
        )

        # 风险评估与轨迹选择
        risk_scores = self.risk_assessment(
            preds.view(B, self.num_modes, -1, 2),
            vector_map,
            self._encode_constraints(constraints)
        )
        selected_idx = torch.argmin(risk_scores, dim=1)
        selected_traj = preds.view(B, self.num_modes, -1, 2)[
            torch.arange(B), selected_idx]

        # 意图预测
        intent_logits = self.intent_predictor(traj_emb)  # [B,6]
        lateral_logits = intent_logits[:, :3]  # 横向3类
        longitudinal_logits = intent_logits[:, 3:]  # 纵向3类

        return {
            'pred_trajectories': preds.view(B, self.num_modes, -1, 2),  # [B,M,T,2]
            'selected_traj': selected_traj,  # [B,T,2]
            'decisions': decisions,
            'constraints': constraints,
            'risk_scores': risk_scores,
            'decision_params': self._encode_constraints(constraints),
            'vector_map': vector_map,
            'lateral_intention': lateral_logits,
            'longitudinal_intention': longitudinal_logits,
            'decision_rationale': decision_rationale,  # 决策解释
            'interaction_matrix': interaction_matrix  # 交互矩阵
        }

    def _gather_all_agent_trajs(self, traj_feat):
        """确保正确收集所有4个角色的轨迹"""
        if isinstance(traj_feat, dict):
            B = traj_feat['input_traj'].size(0)  # 批量大小
            T = traj_feat['input_traj'].size(1)  # 轨迹长度
            device = traj_feat['input_traj'].device

            # 初始化轨迹张量 [B, 4, T, 5]
            agent_trajs = torch.zeros(B, 4, T, 5, device=device)

            # 添加ego车辆轨迹 (角色0)
            agent_trajs[:, 0] = traj_feat['input_traj']

            # 添加其他车辆轨迹
            for i in range(B):
                # 确保每个样本都有3个其他车辆
                others = traj_feat.get('other_trajs', [{} for _ in range(3)])[i]
                for j in range(3):
                    if j < len(others) and 'traj' in others[j]:
                        traj_data = others[j]['traj']
                        # 确保轨迹是张量
                        if not isinstance(traj_data, torch.Tensor):
                            traj_data = torch.tensor(
                                traj_data,
                                dtype=torch.float32,
                                device=device
                            )

                        # 确保轨迹长度匹配
                        if traj_data.size(0) < T:
                            # 填充轨迹
                            padded = torch.zeros(T, 5, device=device)
                            padded[:traj_data.size(0)] = traj_data
                            agent_trajs[i, j + 1] = padded
                        else:
                            agent_trajs[i, j + 1] = traj_data[:T]
                    else:
                        # 填充零值
                        agent_trajs[i, j + 1] = torch.zeros(T, 5, device=device)

            return agent_trajs
        else:
            # 张量输入处理
            B, T, _ = traj_feat.shape
            agent_trajs = torch.zeros(B, 4, T, 5, device=traj_feat.device)
            agent_trajs[:, 0] = traj_feat
            return agent_trajs

    def _get_agent_ids(self, traj_feat, B):
        """获取智能体ID"""
        if 'agent_ids' in traj_feat:
            return traj_feat['agent_ids']
        else:
            # 创建默认ID
            return torch.zeros(B, 4, dtype=torch.long, device=self.device)

    def _encode_constraints(self, constraints):
        """将约束参数编码为张量"""
        return torch.stack([
            torch.tensor([
                c['yaw_range'][0],
                c['yaw_range'][1],
                c['speed_range'][0],
                c['speed_range'][1],
                c['max_lat_acc']
            ], device=self.device) for c in constraints
        ])  # [B,5]

    def _extract_map_features(self, vector_map, index):
        """从批量地图数据中提取单个样本的特征"""
        return {
            'polylines': vector_map['polylines'][index],
            'poly_meta': vector_map['poly_meta'][index]
        }

    def _get_denorm_traj(self, traj_tensor):
        """反标准化轨迹数据"""
        traj_np = traj_tensor.cpu().numpy()
        denorm_traj = np.zeros_like(traj_np)
        denorm_traj[:, :2] = self.scalers['position'].inverse_transform(traj_np[:, :2])
        denorm_traj[:, 3:] = self.scalers['velocity'].inverse_transform(traj_np[:, 3:])
        denorm_traj[:, 2] = traj_np[:, 2]  # yaw保持弧度制
        return denorm_traj

    def _get_current_state(self, traj):
        """获取当前车辆状态"""
        return {
            'yaw': traj[-1, 2].item(),
            'speed': torch.norm(traj[-1, 3:5]).item(),
            'target_yaw': traj[-1, 2].item()
        }

    @property
    def device(self):
        """获取模型所在设备"""
        return next(self.parameters()).device

    def _get_map_info(self, vector_map, index):
        """获取地图元信息"""
        # 检查是否为空地图（Town07）
        if vector_map['polylines'][index].sum() == 0:
            return {
                'curvature': 0.0,
                'lane_type': 4  # 未知类型
            }

        return {
            'curvature': vector_map['poly_meta'][index, :, 1].mean().item(),
            'lane_type': int(vector_map['poly_meta'][index, :, 2].mode().values.item())
        }


class ConstraintAwareDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_steps, num_modes, constraint_dim, vehicle_params):
        super().__init__()
        self.pred_steps = pred_steps
        self.num_modes = num_modes
        self.vehicle_params = vehicle_params
        self.constraint_dim = constraint_dim
        self.hidden_dim = hidden_dim
        self.prev_pred = None  # 用于存储上一个时间步的预测
        self.prev_yaw = None  # 已存在的yaw缓存

        self.lstm = nn.LSTMCell(2 + 64, hidden_dim)
        self.constraint_proj = nn.Sequential(
            nn.Linear(constraint_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.traj_adjust = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 2)
        )
        # 增强参数初始化
        for layer in self.traj_adjust:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.1)

        # 增加输出层的初始化范围
        nn.init.uniform_(self.traj_adjust[-1].weight, -0.1, 0.1)
        nn.init.constant_(self.traj_adjust[-1].bias, 0.0)

    def forward(self, init_state, memory, constraints, teacher_forcing=None):
        h_t, c_t = init_state
        batch_size = constraints.size(0)  # B*M
        device = memory.device

        # ==== 关键修复：添加约束特征计算 ====
        constraint_feat = self.constraint_proj(constraints)  # [B*M, 64]

        # 改进初始状态 - 使用历史轨迹的最后位置
        if teacher_forcing is not None and teacher_forcing.size(1) > 0:
            current_input = teacher_forcing[:, 0].clone().detach()
        else:
            hist_traj = memory[:, -1, :2]  # 假设memory包含历史轨迹
            current_input = hist_traj.clone().detach()

        current_input = torch.clamp(current_input, -10, 10)
        preds = []

        for t in range(self.pred_steps):
            # ==== 关键修复：使用约束特征 ====
            lstm_input = torch.cat([
                current_input,
                constraint_feat  # 直接使用约束特征
            ], dim=1)

            # LSTM前向传播
            h_t, c_t = self.lstm(lstm_input, (h_t, c_t))

            # 注意力机制
            context = self._attention(
                h_t.unsqueeze(1),
                memory
            ).squeeze(1)

            h_t = h_t + context

            # 轨迹预测
            pred = self.traj_adjust(h_t)

            # ==== 关键修复：应用约束 ====
            pred = self._apply_constraints(pred, constraints, t)

            self.prev_pred = pred.detach()

            if teacher_forcing is not None and t < teacher_forcing.size(1):
                current_input = teacher_forcing[:, t] * 0.5 + pred.detach() * 0.5
            else:
                current_input = pred.detach()

            preds.append(pred)

        return torch.stack(preds, dim=1)

    def _apply_constraints(self, pred, constraints, t):
        """改进的约束应用，逐时间步处理"""
        B = pred.size(0)  # 应为B*M

        # 提取约束参数（维度保持[B*M,1]）
        yaw_min = constraints[:, 0].view(-1, 1)  # [B*M,1]
        yaw_max = constraints[:, 1].view(-1, 1)
        speed_min = constraints[:, 2].view(-1, 1)
        speed_max = constraints[:, 3].view(-1, 1)
        max_lat_acc = constraints[:, 4].view(-1, 1)

        # 航向角约束
        yaw_pred = torch.atan2(pred[:, 1], pred[:, 0]).unsqueeze(1)  # [B*M,1]
        yaw_clipped = torch.clamp(yaw_pred, yaw_min, yaw_max)

        # 速度约束
        speed = torch.norm(pred, dim=1, keepdim=True)  # [B*M,1]
        speed_clipped = torch.clamp(speed, speed_min, speed_max)
        pred = pred * (speed_clipped / (speed + 1e-6))

        # 横向加速度约束（使用当前速度）
        if t > 0 and hasattr(self, 'prev_yaw'):  # 需要保存历史信息
            delta_yaw = yaw_pred - self.prev_yaw
            lat_acc = speed * delta_yaw / 0.1  # 时间间隔0.1s
            lat_acc_clamped = torch.clamp(lat_acc, -max_lat_acc, max_lat_acc)
            pred = pred * (1.0 - 0.1 * torch.abs(lat_acc - lat_acc_clamped))
            self.prev_yaw = yaw_pred.detach()
        elif t == 0:
            self.prev_yaw = yaw_pred.detach()

        # 新增速度平滑约束
        if t > 0:
            # 使用上一步保存的预测结果
            prev_speed = torch.norm(self.prev_pred, dim=1, keepdim=True)  # [B*M,1]
            current_speed = torch.norm(pred, dim=1, keepdim=True)
            acc = (current_speed - prev_speed) / 0.1  # 时间间隔0.1s
            acc_clamp = torch.clamp(acc, -6.0, 4.0)
            pred = pred * (acc_clamp / (acc + 1e-6))

        # 增强横向加速度约束（修正参数传递）
        lat_acc = self._calc_lateral_acc(pred.unsqueeze(1))  # [B*M,1]
        lat_acc_clamp = torch.clamp(lat_acc, -3.0, 3.0)
        pred = pred * (lat_acc_clamp.unsqueeze(-1) / (lat_acc.unsqueeze(-1) + 1e-6))

        return pred  # 保持[B*M,2]维度

    def _calc_lateral_acc(self, traj):
        """基于位置变化计算横向加速度"""
        if traj.size(1) < 2:
            return torch.zeros(traj.size(0), device=traj.device)

        # 计算速度向量 (m/s, Δt=0.1s)
        velocity = (traj[:, 1:] - traj[:, :-1]) / 0.1  # [B*M, T-1, 2]
        speed = torch.norm(velocity, dim=-1)  # [B*M, T-1]

        # 计算航向角变化率 (rad/s)
        yaw = torch.atan2(velocity[..., 1], velocity[..., 0])  # [B*M, T-1]
        if yaw.size(1) < 2:
            return torch.zeros(traj.size(0), device=traj.device)
        yaw_diff = torch.diff(yaw, dim=1)  # [B*M, T-2]
        yaw_rate = yaw_diff / 0.1  # [B*M, T-2]

        # 横向加速度 = speed * yaw_rate (需要对齐维度)
        lat_acc = speed[:, :-1] * yaw_rate  # [B*M, T-2]

        # 平均最近的加速度
        if lat_acc.size(1) > 0:
            return lat_acc.mean(dim=1)  # [B*M]
        else:
            return torch.zeros(traj.size(0), device=traj.device)

    def _attention(self, query, memory):
        """维度修正后的注意力计算"""
        # query: [B*M, 1, D]
        # memory: [B*M, 1, D]
        return F.scaled_dot_product_attention(
            query.transpose(0, 1),  # [1, B*M, D]
            memory.transpose(0, 1),
            memory.transpose(0, 1)
        ).transpose(0, 1)  # 恢复为 [B*M, 1, D]


# -------------------- 增强型风险评估模块 --------------------
class EnhancedRiskAssessment(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('road_weights',
                             torch.tensor([1.0, 0.8, 1.2, 1.5, 0.5]))
        self.risk_weights = nn.Parameter(torch.tensor([0.3, 0.5, 0.2]))

    def forward(self, pred_traj, map_data, decision_params):
        """
        输入：
            pred_traj: [B, M, T, 2] 多模态预测轨迹
            map_data: 地图数据
            decision_params: [B, 5] 决策参数张量（yaw_low, yaw_high, speed_low, speed_high, risk_level）
        """
        B, M, T, _ = pred_traj.shape

        # ==== 关键修改：调整决策参数维度 ====
        yaw_low = decision_params[:, 0].view(B, 1, 1)  # [B, 1, 1]
        yaw_high = decision_params[:, 1].view(B, 1, 1)
        speed_low = decision_params[:, 2].view(B, 1, 1)
        speed_high = decision_params[:, 3].view(B, 1, 1)
        risk_level = decision_params[:, 4].long()  # [B]

        # 碰撞风险计算
        road_dist = self._calc_road_distance(pred_traj, map_data)
        collision_risk = torch.sigmoid(5 * (1 - road_dist / 5))  # [B, M, T]

        # 决策偏离风险（修正维度广播）
        yaw = torch.atan2(pred_traj[..., 1], pred_traj[..., 0])  # [B, M, T]
        yaw_violation = torch.clamp(yaw - yaw_high, 0) + torch.clamp(yaw_low - yaw, 0)
        yaw_risk = yaw_violation.mean(dim=-1)  # [B, M]

        speed = torch.norm(pred_traj[..., 3:5], dim=-1)  # [B, M, T]
        speed_violation = torch.clamp(speed - speed_high, 0) + torch.clamp(speed_low - speed, 0)
        speed_risk = speed_violation.mean(dim=-1)  # [B, M]

        # 舒适度风险
        acc = torch.diff(speed, dim=-1)  # [B, M, T-1]
        comfort_risk = torch.mean(torch.abs(acc), dim=-1)  # [B, M]

        # 综合风险计算
        risk_scores = (
                self.risk_weights[0] * collision_risk.mean(dim=-1) +
                self.risk_weights[1] * (yaw_risk + speed_risk) +
                self.risk_weights[2] * comfort_risk
        )

        # 应用风险等级调整（保持维度正确性）
        risk_scores = risk_scores * torch.tensor(
            [0.8, 1.0, 1.2],
            device=risk_scores.device
        )[risk_level].view(B, 1)  # [B, 1]

        return risk_scores

    def _calc_road_distance(self, traj, map_data):
        """基于新地图格式的距离计算"""
        # traj: [B, M, T, 2]
        # map_data: polylines [B, P, N, 6]
        B, M, T, _ = traj.shape
        _, P, N, _ = map_data['polylines'].shape

        # 提取有效道路中心线（类型0为车道中心线）
        road_mask = (map_data['polylines'][..., 4] == 0)  # 类型0为车道中心线
        road_points = map_data['polylines'][..., :2][road_mask]  # [总有效点数, 2]

        # 计算轨迹点到最近道路点的距离
        traj_flat = traj.view(B * M * T, 2)
        dist_matrix = torch.cdist(traj_flat, road_points)  # [B*M*T, 总道路点数]
        min_dists = dist_matrix.min(dim=1)[0].view(B, M, T)

        # 应用道路类型权重
        road_types = map_data['polylines'][..., 4].long()  # [B, P, N]
        # 修复：确保road_weights与road_types在同一设备
        type_weights = self.road_weights.to(road_types.device)[road_types]  # [B, P, N]
        avg_weights = type_weights[road_mask].mean()  # 计算有效道路的平均权重

        return min_dists * avg_weights


# -------------------- 损失函数 --------------------
class ConstraintAwareLoss(nn.Module):
    def __init__(self, num_modes, alpha=1.2, beta=1.0, gamma=0.6,
                 risk_weight=0.1, constraint_alpha=0.3,
                 diversity_weight=0.3):
        super().__init__()
        self.num_modes = num_modes
        self.alpha = alpha  # ADE权重
        self.beta = beta  # FDE权重
        self.gamma = gamma  # 中期预测权重
        self.risk_weight = risk_weight  # 风险损失权重
        self.constraint_alpha = constraint_alpha  # 约束违规权重
        self.diversity_weight = diversity_weight  # 多样性损失权重
        self.log_counter = 0  # 添加日志计数器

        # 基础损失组件
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        self.speed_loss = nn.HuberLoss(delta=0.5, reduction='none')
        self.intent_loss = nn.CrossEntropyLoss()
        self.risk_module = EnhancedRiskAssessment()

    def forward(self, model_outputs, targets, velocity,
                output_valid_len, lateral_labels, longitudinal_labels,
                constraints):
        B, M, T_pred, _ = model_outputs['pred_trajectories'].shape
        T_target = targets.shape[1]
        min_steps = min(T_pred, T_target)
        device = targets.device

        displacement = torch.norm(
            model_outputs['pred_trajectories'][:, :, :min_steps, :] -
            targets[:, None, :min_steps, :],
            dim=-1
        )  # [B, M, T]

        # 生成有效掩码（关键修改）
        valid_lengths = torch.clamp(output_valid_len, max=min_steps)
        mask = torch.zeros(B, min_steps, device=device)
        for i in range(B):
            if valid_lengths[i] > 0:
                mask[i, :valid_lengths[i]] = 1.0

        # ==== ADE计算 ====
        ade = (displacement * mask.unsqueeze(1)).sum(dim=2)  # [B, M]
        valid_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        ade = ade / valid_sum  # [B, M]

        # ==== FDE计算修正 ====
        valid_steps = (output_valid_len - 1).clamp(min=0, max=min_steps - 1)

        # 生成三维索引 [B, M]
        batch_indices = torch.arange(B, device=device)[:, None].expand(-1, M)
        mode_indices = torch.arange(M, device=device)[None, :].expand(B, -1)

        # 使用gather正确获取每个样本的最后有效步
        fde = displacement[batch_indices, mode_indices, valid_steps[:, None].expand(-1, M)]

        # 最佳模态选择
        min_ade_indices = ade.argmin(dim=1)  # [B]
        best_ade = ade[torch.arange(B, device=device), min_ade_indices].mean()
        best_fde = fde[torch.arange(B, device=device), min_ade_indices].mean()

        traj_loss = 0.7 * best_ade + 0.3 * best_fde

        # ==== 速度损失计算（添加有效步数过滤）====
        valid_speed_steps = min(min_steps - 1, velocity.size(1) - 1)
        pred_speed = torch.norm(
            model_outputs['pred_trajectories'][:, :, 1:valid_speed_steps + 1] -
            model_outputs['pred_trajectories'][:, :, :valid_speed_steps],
            dim=-1
        )  # [B, M, T-1]

        gt_speed = torch.norm(velocity[:, :valid_speed_steps, :], dim=-1)  # [B, T-1]
        speed_mask = (torch.arange(valid_speed_steps, device=device)[None, :] <
                      (valid_lengths - 1).clamp(min=0)[:, None]).float()  # [B, T-1]

        speed_loss = (self.speed_loss(
            pred_speed.mean(dim=1),  # 各模态平均
            gt_speed
        ) * speed_mask).sum() / speed_mask.sum().clamp(min=1)

        # ===== 3. 意图分类损失 =====
        lateral_loss = self.intent_loss(
            model_outputs['lateral_intention'],
            lateral_labels
        )
        longitudinal_loss = self.intent_loss(
            model_outputs['longitudinal_intention'],
            longitudinal_labels
        )
        intent_loss = 0.5 * (lateral_loss + longitudinal_loss)

        # ===== 4. 约束违规损失 =====
        constraint_loss = 0
        for i, c in enumerate(constraints):
            # 当前样本所有模态
            pred_traj = model_outputs['pred_trajectories'][i]  # [M,T,2]

            # 航向角违规
            yaw = torch.atan2(pred_traj[..., 1], pred_traj[..., 0])  # [M,T]
            yaw_viol = torch.relu(yaw - c['yaw_range'][1]) + torch.relu(c['yaw_range'][0] - yaw)

            # 速度违规
            speed = torch.norm(pred_traj[:, 1:] - pred_traj[:, :-1], dim=-1) / 0.1  # [M,T-1]
            speed_viol = torch.relu(speed - c['speed_range'][1]) + torch.relu(c['speed_range'][0] - speed)

            # 综合违规
            constraint_loss += (yaw_viol.mean() + speed_viol.mean()) / 2

        constraint_loss /= len(constraints)

        # ===== 5. 多样性损失 =====
        diversity_loss = self._calc_diversity_loss(model_outputs['pred_trajectories'])

        # ===== 6. 风险评估损失 =====
        risk_scores = self.risk_module(
            model_outputs['pred_trajectories'],
            model_outputs['vector_map'],
            model_outputs['decision_params']
        )

        # ===== 总损失计算 =====
        # 调整权重分配，增加轨迹损失比重
        total_loss = (
                0.7 * traj_loss +  # 提高轨迹损失权重
                0.15 * speed_loss +  # 降低速度损失权重
                0.05 * intent_loss +  # 降低意图损失权重
                self.constraint_alpha * constraint_loss +
                self.diversity_weight * diversity_loss +
                self.risk_weight * risk_scores.mean()
        )

        # 记录各项损失值用于分析
        loss_components = {
            'traj': 0.7 * traj_loss.item(),
            'speed': 0.15 * speed_loss.item(),
            'intent': 0.05 * intent_loss.item(),
            'constraint': (self.constraint_alpha * constraint_loss).item(),
            'diversity': (self.diversity_weight * diversity_loss).item(),
            'risk': (self.risk_weight * risk_scores.mean()).item()
        }

        # 每100个batch记录一次损失组件
        if self.log_counter % 100 == 0:
            logger.info(f"损失组件: {loss_components}")
        self.log_counter += 1

        return total_loss

    def _calc_diversity_loss(self, preds):
        """多模态多样性计算"""
        B, M, T, _ = preds.shape
        if M == 1:
            return torch.tensor(0.0, device=preds.device)

        # 计算模态间平均距离
        pred_flat = preds.view(B, M, -1)  # [B,M,T*2]
        pairwise_dist = torch.cdist(pred_flat, pred_flat)  # [B,M,M]

        # 排除自比较
        mask = ~torch.eye(M, dtype=torch.bool, device=preds.device)
        valid_dists = pairwise_dist[:, mask].view(B, M, M - 1)

        # 鼓励各模态保持至少1米差异
        diversity = torch.relu(1.0 - valid_dists.mean(dim=2))
        return diversity.mean()


def train_enhanced():
    # 数据集加载
    train_set = EnhancedAccidentDataset(
        split="train",
        mode='train',
        augment_prob=0.7,
        noise_scale=torch.tensor([0.1, 0.1, 0.05, 0.2, 0.2]))

    # 创建100个样本的子集
    from torch.utils.data import Subset
    full_len = len(train_set)
    sample_indices = torch.randperm(full_len)[:100]
    train_subset = Subset(train_set, sample_indices)

    # 初始化模型
    model = EnhancedHierarchicalPredictor(
        num_modes=5,
        use_llm=True,
        vehicle_params=VEHICLE_PARAMS,
        scalers=train_set.scalers
    ).cuda()

    optimizer = optim.AdamW([
        {'params': model.map_encoder.parameters(), 'lr': 2e-4},
        {'params': model.traj_encoder.parameters(), 'lr': 3e-4},
        {'params': model.risk_assessment.parameters(), 'lr': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 3e-4}
    ], weight_decay=1e-4)

    val_set = EnhancedAccidentDataset(
        split="val",
        mode='val',
        scalers=train_set.scalers)

    full_len = len(val_set)
    sample_indices = torch.randperm(full_len)[:100]
    val_subset = Subset(val_set, sample_indices)

    def collate_fn(batch):
        # 确定最大时间步长
        input_lens = [len(x['input_traj']) for x in batch]
        max_input_len = max(input_lens) if input_lens else 20

        # 轨迹填充
        input_traj = pad_sequence(
            [x['input_traj'] for x in batch],
            batch_first=True,
            padding_value=0.0
        )
        target_traj = pad_sequence(
            [x['target_traj'] for x in batch],
            batch_first=True,
            padding_value=0.0
        ).float()

        # 其他车辆轨迹处理 - 简化结构
        all_other_trajs = []
        agent_ids_list = []

        for b in batch:
            sample_others = []
            # 确保总是有3个其他车辆
            others = b.get('other_trajs', [])
            for j in range(3):
                if j < len(others) and 'traj' in others[j]:
                    traj = others[j]['traj']
                    # 确保轨迹是张量
                    if not isinstance(traj, torch.Tensor):
                        traj = torch.tensor(traj, dtype=torch.float32)

                    # 填充轨迹到最大长度
                    if len(traj) < max_input_len:
                        padded = torch.zeros(max_input_len, 5)
                        padded[:len(traj)] = traj
                        sample_others.append({
                            'id': others[j].get('id', -1),
                            'traj': padded
                        })
                    else:
                        sample_others.append({
                            'id': others[j].get('id', -1),
                            'traj': traj[:max_input_len]
                        })
                else:
                    # 创建占位符
                    sample_others.append({
                        'id': -1,
                        'traj': torch.zeros(max_input_len, 5)
                    })
            all_other_trajs.append(sample_others)
            agent_ids_list.append([b['ego_id'], -1, -1, -1])  # 只保留ego_id

        # 创建批处理后的字典
        collated_batch = {
            'input_traj': input_traj,
            'target_traj': target_traj,
            'velocity': pad_sequence([x['velocity'] for x in batch],
                                     batch_first=True, padding_value=0.0).float(),
            'raw_targets': pad_sequence([b['raw_targets'] for b in batch], batch_first=True),
            'vector_map': {
                'polylines': torch.stack([b['vector_map']['polylines'] for b in batch]),
                'poly_meta': torch.stack([b['vector_map']['poly_meta'] for b in batch])
            },
            'intention_label': torch.stack([b['intention_label'] for b in batch]),
            'scenario_type': torch.tensor([b['scenario_type'] for b in batch]),
            'output_valid_len': torch.tensor([b['output_valid_len'] for b in batch]),
            'other_trajs': all_other_trajs,
            'agent_ids': torch.tensor(agent_ids_list, dtype=torch.long),
            'ego_id': torch.tensor([b['ego_id'] for b in batch]),
            'road_type': torch.tensor([b['road_type'] for b in batch], dtype=torch.long)
        }

        return collated_batch

    train_loader = DataLoader(
        train_subset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
        timeout=300
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=16,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 学习率调度
    total_steps = 5 * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[1e-4, 2e-4, 3e-4, 3e-4],
        total_steps=total_steps,
        pct_start=0.3
    )

    scaler = GradScaler()
    criterion = ConstraintAwareLoss(num_modes=model.num_modes)

    # 训练参数
    total_epochs = 5
    max_pred_steps = 30
    steps_schedule = np.linspace(10, max_pred_steps, total_epochs, dtype=int).tolist()

    for epoch in range(total_epochs):
        if model.decision_processor and epoch == 0:
            logger.info("正在预缓存常见轨迹特征...")
            _precache_common_patterns(model.decision_processor, train_loader)

        current_steps = steps_schedule[epoch]
        model.decoder.pred_steps = current_steps
        tf_prob = max(0.8 - epoch / 500 * 0.7, 0.1)

        # 训练循环
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            vector_map = {
                'polylines': batch['vector_map']['polylines'].cuda(),
                'poly_meta': batch['vector_map']['poly_meta'].cuda()
            }
            trajs = batch['input_traj'].cuda()
            targets = batch['target_traj'][..., :2].cuda()
            velocity = F.avg_pool1d(batch['velocity'].cuda().transpose(1, 2), 3, padding=1).transpose(1, 2)

            teacher_forcing = None
            if torch.rand(1).item() < tf_prob and current_steps > 0:
                teacher_forcing = targets[:, :current_steps]
                teacher_forcing = teacher_forcing.repeat_interleave(model.num_modes, dim=0)
                teacher_forcing += torch.randn_like(teacher_forcing) * 0.05

            with autocast():
                # 构造traj_feat字典
                traj_feat_dict = {
                    'input_traj': trajs,
                    'other_trajs': batch['other_trajs'],
                    'agent_ids': batch['agent_ids'].cuda(),
                    'road_type': batch['road_type'].cuda()  # 确保传递道路类型
                }

                outputs = model(vector_map, traj_feat_dict)  # 现在road_type会正确传递
                pred_trajs = outputs['selected_traj']

                # 获取意图标签 - 关键修改
                lateral_labels = batch['intention_label'][:, 0].long().cuda()
                longitudinal_labels = batch['intention_label'][:, 1].long().cuda()

                # 添加调试信息
                logger.debug(f"输入轨迹形状: {trajs.shape}")
                logger.debug(
                    f"地图数据形状: polylines={vector_map['polylines'].shape}, poly_meta={vector_map['poly_meta'].shape}")
                logger.debug(f"其他车辆数量: {len(traj_feat_dict['other_trajs'])}")

                # 添加调试信息
                if 'interaction_matrix' in outputs:
                    logger.debug(f"交互矩阵形状: {outputs['interaction_matrix'].shape}")

                # 计算损失
                loss = criterion(
                    model_outputs=outputs,
                    targets=targets,
                    velocity=velocity,
                    output_valid_len=batch['output_valid_len'].cuda(),
                    lateral_labels=lateral_labels,
                    longitudinal_labels=longitudinal_labels,
                    constraints=outputs['constraints']
                )

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 日志记录
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")
                if model.decision_processor:
                    model.decision_processor.save_cache()
                    logger.info(f"LLM缓存大小: {len(model.decision_processor.cache)}")

            # 每50个batch记录详细数据统计
            if batch_idx % 50 == 0:
                # 轨迹统计
                input_mean = trajs.mean().item()
                input_std = trajs.std().item()
                pred_mean = pred_trajs.mean().item()
                pred_std = pred_trajs.std().item()

                # 梯度统计
                grad_norms = [
                    p.grad.norm().item() if p.grad is not None else 0
                    for p in model.parameters()
                ]
                avg_grad_norm = sum(grad_norms) / len(grad_norms)

                logger.info(
                    f"Epoch {epoch} Batch {batch_idx} | "
                    f"Input Mean: {input_mean:.4f} ± {input_std:.4f} | "
                    f"Pred Mean: {pred_mean:.4f} ± {pred_std:.4f} | "
                    f"Grad Norm: {avg_grad_norm:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

        # 日志记录（修正位置）
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f} | "
                    f"ADE: {outputs['risk_scores'].mean().item():.2f} | "
                    f"FDE: {outputs['risk_scores'].max().item():.2f}"
                )

        # 验证与保存
        if epoch % 2 == 0:
            model_dir = os.path.expanduser('~/trained_models')
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'llm_cache': model.decision_processor.cache if model.decision_processor else None
            }, os.path.join(model_dir, f'model_epoch_{epoch}.pth'))

            model.decoder.pred_steps = 30
            validate_enhanced(model, val_loader,
                              train_set.scalers['position'],
                              train_set.scalers['velocity'])
            model.decoder.pred_steps = current_steps


def _precache_common_patterns(processor, loader):
    """预缓存常见轨迹模式"""
    for batch in loader:
        trajs = batch['input_traj'].numpy()
        for traj in trajs:
            # 将numpy数组转换为Python列表
            features = {
                "traj": traj.tolist()  # 关键修改：使用tolist()转换numpy数组
            }
            sorted_features = json.dumps(features, sort_keys=True, separators=(',', ':'))
            cache_key = hashlib.sha256(sorted_features.encode()).hexdigest()
            if cache_key not in processor.cache:
                processor.process(features)  # 传递处理后的字典


def validate_enhanced(model, val_loader, position_scaler, velocity_scaler):
    model.eval()
    llm_decisions = defaultdict(list)
    metrics = defaultdict(lambda: {'ade': [], 'fde': []})
    constraint_stats = defaultdict(list)
    skipped_samples = 0  # 跟踪跳过的样本数

    with torch.no_grad():
        for batch in val_loader:
            B = batch['scenario_type'].shape[0]
            vector_map = {
                'polylines': batch['vector_map']['polylines'].cuda(),
                'poly_meta': batch['vector_map']['poly_meta'].cuda()
            }
            trajs = batch['input_traj'].cuda()
            raw_targets = batch['raw_targets'].numpy()  # 原始未标准化目标轨迹
            scenarios = batch['scenario_type'].numpy()
            output_valid_len = batch['output_valid_len'].numpy()

            # 模型推理
            traj_feat_dict = {
                'input_traj': trajs,
                'other_trajs': batch['other_trajs'],
                'agent_ids': batch['agent_ids'].cuda(),
                'road_type': batch['road_type'].cuda()
            }
            outputs = model(vector_map, traj_feat_dict)
            constraints = outputs['constraints']
            selected_traj = outputs['selected_traj']

            preds_np = selected_traj.cpu().numpy()

            # ==== 关键修改：添加空数组检查 ====
            # 反标准化位置
            position_points = preds_np[..., :2].reshape(-1, 2)
            if position_points.shape[0] > 0:  # 检查是否有数据点
                preds_position = position_scaler.inverse_transform(position_points).reshape(B, -1, 2)
            else:
                preds_position = np.zeros((B, preds_np.shape[1], 2))  # 创建空数组
                logger.warning(f"位置反标准化: 遇到空数组, 创建零数组替代")

            # 反标准化速度
            velocity_points = preds_np[..., 3:5].reshape(-1, 2)
            if velocity_points.shape[0] > 0:  # 检查是否有数据点
                preds_velocity = velocity_scaler.inverse_transform(velocity_points).reshape(B, -1, 2)
            else:
                preds_velocity = np.zeros((B, preds_np.shape[1], 2))  # 创建空数组
                logger.warning(f"速度反标准化: 遇到空数组, 创建零数组替代")

            # 反标准化真实轨迹
            target_position_points = batch['target_traj'].numpy().reshape(-1, 2)
            if target_position_points.shape[0] > 0:
                targets_position = position_scaler.inverse_transform(target_position_points).reshape(B, -1, 2)
            else:
                targets_position = np.zeros_like(batch['target_traj'].numpy())

            # 计算速度
            pred_speed = np.linalg.norm(preds_velocity, axis=-1)
            gt_speed = np.linalg.norm(batch['velocity'].numpy(), axis=-1)

            # 约束检查
            for i in range(B):
                if i >= len(constraints) or not constraints[i]:
                    continue

                c = constraints[i]
                valid_steps = min(preds_position.shape[1], int(output_valid_len[i]))

                # 只处理有有效步长的样本
                if valid_steps > 0:
                    # 航向角计算 (使用位置差分)
                    if valid_steps > 1:
                        displacements = preds_position[i, 1:valid_steps] - preds_position[i, :valid_steps - 1]
                        pred_yaw = np.arctan2(displacements[:, 1], displacements[:, 0])
                    else:
                        pred_yaw = np.array([0.0])

                    # 航向约束检查
                    yaw_in = (pred_yaw >= c['yaw_range'][0]) & (pred_yaw <= c['yaw_range'][1])
                    yaw_ratio = yaw_in.mean()

                    # 速度约束检查
                    speed_in = (pred_speed[i, :valid_steps] >= c['speed_range'][0]) & \
                               (pred_speed[i, :valid_steps] <= c['speed_range'][1])
                    speed_ratio = speed_in.mean()

                    constraint_stats['yaw'].append(yaw_ratio)
                    constraint_stats['speed'].append(speed_ratio)

            # 指标计算 (只处理有效样本)
            for i in range(B):
                actual_len = max(0, min(
                    int(output_valid_len[i]),
                    preds_position.shape[1],
                    targets_position.shape[1]
                ))

                if actual_len == 0:
                    skipped_samples += 1
                    continue

                # 位置误差
                pos_error = np.linalg.norm(
                    preds_position[i, :actual_len] - targets_position[i, :actual_len],
                    axis=1
                )

                # 速度误差 (只比较有效步长)
                valid_speed_steps = min(actual_len, gt_speed.shape[1])
                speed_error = np.abs(
                    pred_speed[i, :valid_speed_steps] - gt_speed[i, :valid_speed_steps]
                )

                # 综合ADE (位置+速度)
                ade = 0.7 * np.mean(pos_error) + 0.3 * np.mean(speed_error)

                # 综合FDE (最终位置+最终速度)
                fde = 0.7 * pos_error[-1] + 0.3 * speed_error[-1] if valid_speed_steps > 0 else pos_error[-1]

                scenario = 'accident' if scenarios[i] else 'normal'
                metrics[scenario]['ade'].append(ade)
                metrics[scenario]['fde'].append(fde)

                # 记录LLM决策
                if model.decision_processor and i < len(outputs['decisions']):
                    decision = outputs['decisions'][i]
                    key = f"{decision['path_decision']}|{decision['speed_decision']}"
                    llm_decisions[key].append({
                        'scenario': scenario,
                        'ade': ade,
                        'fde': fde
                    })

    # 结果报告
    logger.info(f"跳过 {skipped_samples} 个无效样本 (有效长度=0)")

    # ==== 结果打印 ====
    logger.info("\n================ 验证结果分析 ================")

    # 场景指标
    for scenario in ['accident', 'normal']:
        if not metrics[scenario]['ade']:
            continue

        ade_vals = np.array(metrics[scenario]['ade'])
        fde_vals = np.array(metrics[scenario]['fde'])

        logger.info(f"\n🔍 {scenario.upper()} 场景 (样本数: {len(ade_vals)})")
        logger.info(f"  ADE: {ade_vals.mean():.2f}m ± {ade_vals.std():.2f}")
        logger.info(f"  FDE: {fde_vals.mean():.2f}m ± {fde_vals.std():.2f}")

    # 约束遵守率
    if constraint_stats:
        logger.info("\n🚦 约束遵守情况:")
        logger.info(f"  平均偏航角遵守率: {np.mean(constraint_stats['yaw']):.2%}")
        logger.info(f"  平均速度遵守率: {np.mean(constraint_stats['speed']):.2%}")
    else:
        logger.warning("未找到有效约束数据")

    # LLM决策分析
    if llm_decisions:
        logger.info("\n🧠 LLM决策分布:")
        for decision, records in llm_decisions.items():
            accident_count = sum(1 for r in records if r['scenario'] == 'accident')
            total = len(records)
            avg_violation = {
                'yaw': np.mean([r['constraint_violation']['yaw'] for r in records]),
                'speed': np.mean([r['constraint_violation']['speed'] for r in records])
            }

            logger.info(
                f"  {decision}: "
                f"出现次数: {total} | "
                f"事故占比: {accident_count / total:.1%} | "
                f"偏航违规率: {avg_violation['yaw']:.1%} | "
                f"速度违规率: {avg_violation['speed']:.1%}"
            )

    return metrics


if __name__ == "__main__":
    train_enhanced()