import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mmcv
from mmengine.fileio import load
from tqdm import tqdm
from collections import defaultdict


# ==================== 1. 数据解析模块 ====================
def parse_meta_file(meta_path):
    """解析meta文件提取Agent ID和场景信息"""
    agent_ids = {}
    with open(meta_path, 'r') as f:
        content = f.read()

        # 提取agents id
        id_match = re.search(r'agents id:\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)', content)
        if id_match:
            ids = list(map(int, id_match.groups()))
            agent_ids = {
                "ego": ids[0],
                "ego_behind": ids[1],
                "other": ids[2],
                "other_behind": ids[3]
            }

        # 提取道路信息
        road_match = re.search(r'road_type:\s*(.+)', content)
        road_type = road_match.group(1) if road_match else "unknown"

        # 提取车辆方向
        directions = {}
        ego_dir = re.search(r'ego_vehicle_direction:\s*(\w+)', content)
        other_dir = re.search(r'other_vehicle_direction:\s*(\w+)', content)
        if ego_dir: directions["ego"] = ego_dir.group(1)
        if other_dir: directions["other"] = other_dir.group(1)

    return agent_ids, road_type, directions


def parse_label_file(label_path):
    """解析label文件中的轨迹信息 - 修正Y轴反转问题"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if not lines or len(lines) < 2:
            return [], []

        # 第一行是自车速度
        vehicle_speed = list(map(float, lines[0].split()))
        trajectories = []

        # 后续行是其他车辆信息
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 11:
                continue

            try:
                # 确保有足够的字段
                if len(parts) < 11:
                    continue

                # 提取车辆ID（倒数第三列）
                vehicle_id = int(parts[-3])

                # 提取位置、尺寸、方向、速度 - 移除不必要的反转
                position = list(map(float, parts[1:4]))  # x, y, z
                dimensions = list(map(float, parts[4:7]))  # l, w, h
                yaw = float(parts[7])  # 不再反转yaw角

                # 速度向量处理 - 不再反转Y分量
                velocity = list(map(float, parts[8:10]))  # vx, vy

                num_lidar_pts = int(parts[-2])
                camera_visibility = parts[-1] == "True"

                trajectories.append({
                    'id': vehicle_id,
                    'position': position,
                    'dimensions': dimensions,
                    'yaw': yaw,
                    'velocity': velocity,
                    'num_lidar_pts': num_lidar_pts,
                    'camera_visibility': camera_visibility
                })
            except (ValueError, IndexError, TypeError) as e:
                print(f"解析错误 {label_path}: {e}, 行内容: {line}")
                continue

        return vehicle_speed, trajectories


def transform_to_world(local_point, ego_to_world_matrix):
    """将本地坐标转换为世界坐标 - 移除不必要的反转"""
    point = np.array([local_point[0], local_point[1], local_point[2], 1.0])
    world_point = np.dot(ego_to_world_matrix, point)
    return world_point[:3]  # 直接返回转换结果


def get_vehicle_trajectories(scenario_path, agent_ids, scene_name):
    """获取所有车辆的完整轨迹（世界坐标系）- 统一使用自车变换矩阵"""
    # 只使用自车视角
    vehicle_type = "ego_vehicle"
    role = "ego"
    vehicle_id = agent_ids[role]

    all_trajectories = defaultdict(list)
    frame_data = defaultdict(dict)

    print(f"提取轨迹数据: {scene_name} (统一使用自车视角)")

    # Y轴反转矩阵（与数据生成代码一致）
    y_reverse_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    label_dir = os.path.join(scenario_path, vehicle_type, "label", scene_name)
    calib_dir = os.path.join(scenario_path, vehicle_type, "calib", scene_name)

    # 检查目录是否存在
    if not os.path.exists(label_dir):
        print(f"警告: 缺少标签目录 {label_dir}")
        return dict(all_trajectories), frame_data
    if not os.path.exists(calib_dir):
        print(f"警告: 缺少标定目录 {calib_dir}")
        return dict(all_trajectories), frame_data

    # 获取所有时间步
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    if not label_files:
        print(f"警告: 未找到标签文件 {label_dir}")
        return dict(all_trajectories), frame_data

    print(f"处理 {vehicle_type}, 找到 {len(label_files)} 个标签文件")

    for label_file in tqdm(label_files, desc=f"处理 {vehicle_type} 标签"):
        # 从文件名中提取时间步
        filename = os.path.basename(label_file)
        try:
            frame_num = int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            print(f"文件名解析错误: {filename}")
            continue

        # 加载标定数据 (统一使用自车标定)
        calib_file = os.path.join(calib_dir, f"{scene_name}_{frame_num:03d}.pkl")
        if not os.path.exists(calib_file):
            # 尝试其他可能的命名
            alt_calib_file = os.path.join(calib_dir, filename.replace('.txt', '.pkl'))
            if os.path.exists(alt_calib_file):
                calib_file = alt_calib_file
            else:
                base_name = filename.split('.')[0]
                alt_calib_file2 = os.path.join(calib_dir, f"{base_name}.pkl")
                if os.path.exists(alt_calib_file2):
                    calib_file = alt_calib_file2
                else:
                    print(f"警告: 未找到标定文件 {calib_file}")
                    continue

        try:
            # 加载标定数据
            calib_data = load(calib_file)
            if "ego_to_world" not in calib_data:
                print(f"标定文件缺少 ego_to_world 矩阵: {calib_file}")
                continue

            # 应用Y轴反转矩阵（与数据生成代码一致）
            ego_to_world = y_reverse_matrix @ calib_data["ego_to_world"]

            # 解析标签文件
            _, frame_trajectories = parse_label_file(label_file)
            if not frame_trajectories:
                print(f"警告: 未解析到轨迹数据 {label_file}")
                continue

            # 处理所有车辆轨迹 (统一使用自车变换矩阵)
            for traj in frame_trajectories:
                # 当前车辆自身 (自车)
                if traj["id"] == -100:
                    world_pos = transform_to_world(traj['position'], ego_to_world)
                    traj_data = {
                        'frame': frame_num,
                        'position': world_pos.tolist(),
                        'velocity': traj['velocity'],
                        'yaw': traj['yaw'],
                        'role': role,
                        'vehicle_type': vehicle_type,
                        'ego_to_world_matrix': ego_to_world.tolist()  # 保存变换矩阵
                    }
                    all_trajectories[vehicle_id].append(traj_data)
                    frame_data[frame_num][vehicle_id] = traj_data
                # 其他已知车辆
                elif traj["id"] in agent_ids.values():
                    # 将其他车辆的局部坐标转换为世界坐标 (使用自车变换矩阵)
                    world_pos = transform_to_world(traj['position'], ego_to_world)
                    traj_data = {
                        'frame': frame_num,
                        'position': world_pos.tolist(),
                        'velocity': traj['velocity'],
                        'yaw': traj['yaw'],
                        'id': traj['id'],
                        'role': next((k for k, v in agent_ids.items() if v == traj['id']), "unknown"),
                        'vehicle_type': vehicle_type,
                        'ego_to_world_matrix': ego_to_world.tolist()  # 保存变换矩阵
                    }
                    all_trajectories[traj['id']].append(traj_data)
                    frame_data[frame_num][traj['id']] = traj_data

        except Exception as e:
            print(f"处理文件错误 {label_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 按时间排序
    for vehicle_id in all_trajectories:
        all_trajectories[vehicle_id].sort(key=lambda x: x['frame'])

    # 打印轨迹统计信息
    print(f"提取的轨迹统计:")
    for vehicle_id, points in all_trajectories.items():
        role = points[0]['role'] if points else "unknown"
        print(f"  Vehicle ID: {vehicle_id}, 角色: {role}, 轨迹点数: {len(points)}")
        if points:
            first_point = points[0]
            last_point = points[-1]
            print(f"    起始帧: {first_point['frame']}, 位置: {first_point['position']}")
            print(f"    结束帧: {last_point['frame']}, 位置: {last_point['position']}")

    return dict(all_trajectories), frame_data


# ==================== 2. 地图构建模块 ====================
class EnhancedBEVMapExtractor:
    def __init__(self, map_range=200.0):
        """增强版BEV地图提取器"""
        # Carla语义类映射
        self.class_mapping = {
            0: "Unlabeled", 1: "Building", 2: "Fence", 3: "Other", 4: "Pedestrian",
            5: "Pole", 6: "RoadLine", 7: "Road", 8: "SideWalk", 9: "Vegetation",
            10: "Vehicles", 11: "Wall", 12: "TrafficSign", 13: "Sky", 14: "Ground",
            15: "Bridge", 16: "RailTrack", 17: "GuardRail", 18: "TrafficLight",
            19: "Static", 20: "Dynamic", 21: "Water", 22: "Terrain"
        }

        # 定义可视化颜色映射
        self.color_map = {
            0: [0, 0, 0], 1: [70, 70, 70], 2: [100, 40, 40], 3: [55, 90, 80],
            4: [220, 20, 60], 5: [153, 153, 153], 6: [157, 234, 50], 7: [128, 64, 128],
            8: [244, 35, 232], 9: [107, 142, 35], 10: [0, 0, 142], 11: [102, 102, 156],
            12: [220, 220, 0], 13: [70, 130, 180], 14: [81, 0, 81], 15: [150, 100, 100],
            16: [230, 150, 140], 17: [180, 165, 180], 18: [250, 170, 30], 19: [110, 190, 160],
            20: [170, 120, 50], 21: [45, 60, 150], 22: [145, 170, 100]
        }

        # 地图参数
        self.map_range = map_range
        self.resolution = map_range / 1200  # 每像素代表多少米
        self.image_size = 1200

    def load_bev_file(self, file_path):
        """加载BEV语义分割NPZ文件"""
        try:
            data = np.load(file_path)
            # 尝试不同的键名
            for key in ['bev_instance', 'data', 'arr_0']:
                if key in data:
                    bev_array = data[key]
                    break
            else:
                # 如果没有找到标准键，取第一个数组
                bev_array = data[list(data.keys())[0]]

            # 调整数组形状
            if bev_array.ndim == 2:
                # 如果是二维数组，添加第三维
                bev_array = np.stack([bev_array] * 3, axis=-1)
            elif bev_array.shape[2] > 3:
                # 如果通道数超过3，取前3个通道
                bev_array = bev_array[:, :, :3]

            # 调整大小
            if bev_array.shape[0] != self.image_size or bev_array.shape[1] != self.image_size:
                bev_array = cv2.resize(bev_array, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            return bev_array
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None

    def extract_class_map(self, bev_array):
        """提取语义类别地图"""
        # 使用第三个通道作为类别信息
        return bev_array[:, :, 2].astype(np.uint8)

    def world_to_pixel(self, world_pos, ref_ego_to_world_matrix):
        """
        将轨迹点世界坐标转换为BEV像素坐标（统一参考系）
        参数:
            world_pos: 世界坐标系中的位置 [x, y, z]
            ref_ego_to_world_matrix: 参考帧的自车到世界的变换矩阵 (4x4)
        """
        try:
            # 将世界坐标转换到参考帧的自车坐标系
            world_point = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
            point_in_ref_ego = np.linalg.inv(ref_ego_to_world_matrix) @ world_point

            # 提取坐标值 (注意: BEV坐标系定义)
            # x_ref: 自车前方为正（图像上方）
            # y_ref: 自车左侧为正（图像右侧）
            x_ref = point_in_ref_ego[0]
            y_ref = point_in_ref_ego[1]

            # 计算像素坐标 (统一参考系)
            # BEV图像中心(600,600)对应自车位置
            # 转换公式:
            #   px = (y_ref + 地图半径) / 分辨率  -> 将y_ref映射到图像x轴
            #   py = (地图半径 - x_ref) / 分辨率  -> 将x_ref映射到图像y轴
            px = int((y_ref + self.map_range / 2) / self.resolution)
            py = int((self.map_range / 2 - x_ref) / self.resolution)

            return px, py
        except Exception as e:
            print(f"坐标转换错误: {e}")
            # 返回中心点作为容错
            center = int(self.image_size / 2)
            return center, center



    def extract_static_map(self, bev_files, frame_index=20):
        """从BEV图像中提取静态地图 - 只使用指定帧"""
        # 检查帧索引是否在有效范围内
        if frame_index < 0 or frame_index >= len(bev_files):
            print(f"警告: 请求的帧索引 {frame_index} 超出范围 (0-{len(bev_files) - 1})，使用第0帧代替")
            frame_index = 0

        # 只加载指定帧的BEV文件
        frame_file = bev_files[frame_index]
        print(f"使用第 {frame_index} 帧的BEV文件: {frame_file}")

        # 加载BEV数据
        bev_array = self.load_bev_file(frame_file)
        if bev_array is None:
            return None

        # 提取语义类别地图
        class_map = self.extract_class_map(bev_array)

        # 创建静态地图
        static_map = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # 直接使用单帧数据创建地图
        for i in tqdm(range(self.image_size), desc="生成静态地图"):
            for j in range(self.image_size):
                class_id = class_map[i, j]
                static_map[i, j] = self.color_map.get(class_id, [240, 240, 240])

        # 应用形态学操作增强地图质量
        static_map = self.enhance_map_quality(static_map)
        return static_map

    def enhance_map_quality(self, static_map):
        """应用形态学操作增强地图质量 - 优化版"""
        # 转换为灰度图进行处理
        gray = cv2.cvtColor(static_map, cv2.COLOR_RGB2GRAY)

        # 创建道路和车道线的掩码
        road_mask = np.zeros_like(gray)
        lane_mask = np.zeros_like(gray)

        # 提取道路 (类别7) 和车道线 (类别6)
        for i in range(static_map.shape[0]):
            for j in range(static_map.shape[1]):
                pixel = static_map[i, j]
                if np.array_equal(pixel, self.color_map[7]):  # 道路
                    road_mask[i, j] = 255
                elif np.array_equal(pixel, self.color_map[6]):  # 车道线
                    lane_mask[i, j] = 255

        # 对道路掩码应用形态学操作 - 增强操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 填充道路区域中的小孔洞
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 对车道线掩码应用形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 去除小的噪声区域
        lane_mask = self.remove_small_objects(lane_mask, min_size=50)

        # 创建增强后的地图
        enhanced_map = np.full((self.image_size, self.image_size, 3), 240, dtype=np.uint8)  # 浅灰色背景

        # 应用道路和车道线
        enhanced_map[road_mask > 0] = self.color_map[7]  # 道路颜色
        enhanced_map[lane_mask > 0] = self.color_map[6]  # 车道线颜色

        # 添加道路边缘
        road_edges = cv2.Canny(road_mask, 100, 200)
        enhanced_map[road_edges > 0] = [50, 50, 50]  # 深灰色边缘

        return enhanced_map

    def remove_small_objects(self, mask, min_size=50):
        """去除小的连通区域"""
        # 查找所有连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # 创建一个新的掩码，只保留大于min_size的区域
        new_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                new_mask[labels == i] = 255

        return new_mask

    def visualize_selected_classes(self, class_map, selected_class_ids, save_path=None,
                                   smooth=True, min_keep_area=1500):
        """可视化选定的类别 - 优化版"""
        h, w = class_map.shape
        color_image = np.full((h, w, 3), 240, dtype=np.uint8)  # 浅灰色背景

        # 1. 合成目标区域掩码
        mask = np.isin(class_map, selected_class_ids).astype(np.uint8)

        if smooth:
            # 2. 闭运算填补小孔洞，平滑边缘
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # 3. 开运算去除小噪点
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 4. 连通域分析去除小区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cleaned_mask = np.zeros_like(mask)
            for i in range(1, num_labels):  # 0是背景
                if stats[i, cv2.CC_STAT_AREA] > min_keep_area:
                    cleaned_mask[labels == i] = 1
            mask = cleaned_mask

        # 5. 只在目标mask上着色
        for class_id in selected_class_ids:
            if class_id not in self.color_map:
                continue
            color = self.color_map[class_id]
            class_mask = (class_map == class_id) & (mask == 1)
            color_image[class_mask] = color

        return color_image

    def extract_specific_class(self, class_map, target_class_id, min_area=0):
        """
        提取特定类别的区域

        参数:
            class_map: 语义类别地图，形状为(1200, 1200)
            target_class_id: 目标类别的ID
            min_area: 最小区域大小，小于此面积的区域将被过滤掉

        返回:
            形状为(1200, 1200)的二进制掩码，目标类别区域为1，其他为0
        """
        # 创建目标类别的掩码
        mask = (class_map == target_class_id).astype(np.uint8)

        # 如果需要，应用形态学操作清理掩码
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 如果指定了最小面积，过滤小区域
        if min_area > 0:
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 创建新的掩码
            filtered_mask = np.zeros_like(mask)

            # 保留面积大于阈值的轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    cv2.drawContours(filtered_mask, [contour], -1, 1, -1)

            mask = filtered_mask

        return mask

    def calculate_statistics(self, class_map):
        """
        计算地图统计信息

        参数:
            class_map: 语义类别地图，形状为(1200, 1200)

        返回:
            包含统计信息的字典
        """
        # 计算每个类别的像素数量
        class_counts = {}
        for class_id in np.unique(class_map):
            class_name = self.class_mapping.get(class_id, f"Unknown ({class_id})")
            pixel_count = np.sum(class_map == class_id)
            class_counts[class_name] = pixel_count

        # 计算总面积（像素数）
        total_area = class_map.size

        # 计算每个类别的百分比
        class_percentages = {name: (count / total_area) * 100
                             for name, count in class_counts.items()}

        # 计算地图覆盖率（非零像素的百分比）
        coverage = (np.sum(class_map > 0) / total_area) * 100

        stats = {
            "total_classes": len(np.unique(class_map)),
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "coverage_percentage": coverage
        }

        return stats

    def add_trajectories_to_map(self, static_map, trajectories, frame_data=None,
                                highlight_frame=None, ref_ego_to_world_matrix=None):
        """在静态地图上添加车辆轨迹 - 统一参考系"""
        if ref_ego_to_world_matrix is None:
            print("警告: 未提供参考帧的变换矩阵")
            ref_ego_to_world_matrix = np.eye(4)  # 使用单位矩阵作为默认值

        trajectory_map = static_map.copy()

        # 为每个车辆分配颜色
        vehicle_colors = {}
        vehicle_ids = list(trajectories.keys())
        for i, vehicle_id in enumerate(vehicle_ids):
            hue = i * 180 // max(1, len(vehicle_ids))  # 避免除以零
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            vehicle_colors[vehicle_id] = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        # 绘制所有车辆的轨迹
        for vehicle_id, points in trajectories.items():
            if not points:
                continue

            color = vehicle_colors[vehicle_id]
            prev_point = None

            # 绘制轨迹点和连接线
            for point in points:
                try:
                    # 统一使用参考矩阵转换 (不再使用点自身的变换矩阵)
                    px, py = self.world_to_pixel(
                        point['position'],
                        ref_ego_to_world_matrix  # 统一使用参考帧变换矩阵
                    )

                    # 调试信息 - 打印车辆位置
                    if point['frame'] == highlight_frame:
                        print(f"车辆 {vehicle_id} 位置 - 世界坐标: {point['position']}")
                        print(f"转换后像素坐标: ({px}, {py})")

                    if 0 <= px < self.image_size and 0 <= py < self.image_size:
                        radius = 5 if point.get('highlight', False) else 3
                        cv2.circle(trajectory_map, (px, py), radius, color, -1)
                        if prev_point is not None:
                            prev_px, prev_py = self.world_to_pixel(
                                prev_point['position'],
                                ref_ego_to_world_matrix  # 统一使用参考帧变换矩阵
                            )
                            if (0 <= prev_px < self.image_size and 0 <= prev_py < self.image_size and
                                    0 <= px < self.image_size and 0 <= py < self.image_size):
                                cv2.line(trajectory_map, (prev_px, prev_py), (px, py), color, 3)
                    prev_point = point
                except Exception as e:
                    print(f"轨迹点绘制错误: {e}")
                    continue

        # 高亮显示特定帧
        if highlight_frame is not None and frame_data is not None:
            frame_info = frame_data.get(highlight_frame)
            if frame_info:
                for vehicle_id, data in frame_info.items():
                    color = vehicle_colors.get(vehicle_id, (0, 0, 255))
                    try:
                        px, py = self.world_to_pixel(
                            data['position'],
                            ref_ego_to_world_matrix  # 统一使用参考帧变换矩阵
                        )
                        if 0 <= px < self.image_size and 0 <= py < self.image_size:
                            size = 20
                            cv2.rectangle(trajectory_map, (px - size, py - size),
                                          (px + size, py + size), color, 4)
                            cv2.putText(trajectory_map, f"ID:{vehicle_id}", (px + size + 5, py),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
                            cv2.putText(trajectory_map, f"ID:{vehicle_id}", (px + size + 5, py),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    except Exception as e:
                        print(f"高亮帧绘制错误: {e}")
                        continue

        return trajectory_map

    def create_topological_map(self, road_type, directions):
        """创建拓扑地图"""
        topology = {
            "road_type": road_type,
            "junctions": [],
            "agent_directions": directions
        }

        if "junction" in road_type.lower():
            topology["junctions"].append({
                "type": road_type.split()[0],
                "location": "center"
            })

        return topology


# ==================== 3. 主处理流程 ====================
def process_scenario(scenario_path, scene_name, output_dir, highlight_frame=None):
    """处理单个场景"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 50}")
    print(f"处理场景: {scene_name}")
    print(f"{'=' * 50}")

    # 1. 解析meta文件
    meta_path = os.path.join(scenario_path, "meta", f"{scene_name}.txt")
    if not os.path.exists(meta_path):
        print(f"警告: 未找到meta文件 {meta_path}")
        return

    agent_ids, road_type, directions = parse_meta_file(meta_path)
    print(f"Agent IDs: {agent_ids}")

    # 2. 提取车辆轨迹
    trajectories, frame_data = get_vehicle_trajectories(scenario_path, agent_ids, scene_name)

    if not trajectories:
        print(f"警告: 未提取到轨迹数据")
        return
    else:
        print(f"成功提取 {len(trajectories)} 辆车的轨迹")

    # 保存轨迹数据
    traj_path = os.path.join(output_dir, f"{scene_name}_trajectories.json")
    with open(traj_path, 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f"轨迹数据已保存至: {traj_path}")

    # 3. 初始化地图提取器
    map_extractor = EnhancedBEVMapExtractor(map_range=200.0)

    # 4. 获取BEV文件 - 使用ego_vehicle的视角
    bev_dir = os.path.join(scenario_path, "ego_vehicle", "BEV_instance_camera", scene_name)
    if not os.path.exists(bev_dir):
        print(f"警告: 未找到BEV目录 {bev_dir}")
        return

    bev_files = sorted(glob.glob(os.path.join(bev_dir, "*.npz")))
    if not bev_files:
        print(f"警告: 未找到BEV文件")
        return
    else:
        print(f"找到 {len(bev_files)} 个BEV文件")

    # 5. 构建静态地图 (使用第20帧)
    frame_index = 20
    if frame_index >= len(bev_files):
        frame_index = len(bev_files) - 1
        print(f"警告: 请求的第20帧不存在，使用最后一帧代替 (索引: {frame_index})")

    # 加载BEV文件
    bev_file = bev_files[frame_index]
    bev_array = map_extractor.load_bev_file(bev_file)
    if bev_array is None:
        return None

    # 提取语义类别地图
    class_map = map_extractor.extract_class_map(bev_array)

    # 生成静态地图 - 使用选定的类别
    selected_classes = [6, 7]  # 车道线和道路
    static_map = map_extractor.visualize_selected_classes(
        class_map,
        selected_classes,
        smooth=True,
        min_keep_area=1500
    )

    # 6. 添加轨迹到地图
    if highlight_frame is not None:
        print(f"高亮显示帧: {highlight_frame}")
        for vehicle_id, points in trajectories.items():
            for point in points:
                if point['frame'] == highlight_frame:
                    point['highlight'] = True

    # 获取静态地图对应的变换矩阵
    calib_file = os.path.join(scenario_path, "ego_vehicle", "calib", scene_name,
                              f"{scene_name}_{frame_index:03d}.pkl")
    calib_data = load(calib_file)

    # 应用Y轴反转矩阵（与数据生成代码一致）
    y_reverse_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    ref_ego_to_world = y_reverse_matrix @ calib_data["ego_to_world"]

    # 添加轨迹时传入参考矩阵
    trajectory_map = map_extractor.add_trajectories_to_map(
        static_map,
        trajectories,
        frame_data,
        highlight_frame,
        ref_ego_to_world  # 传入参考矩阵
    )

    # 7. 保存地图
    map_path = os.path.join(output_dir, f"{scene_name}_map_with_trajectories.png")
    cv2.imwrite(map_path, cv2.cvtColor(trajectory_map, cv2.COLOR_RGB2BGR))
    print(f"轨迹地图已保存至: {map_path}")

    # 8. 创建拓扑地图
    topology = map_extractor.create_topological_map(road_type, directions)
    topo_path = os.path.join(output_dir, f"{scene_name}_topology.json")
    with open(topo_path, 'w') as f:
        json.dump(topology, f, indent=2)
    print(f"拓扑地图已保存至: {topo_path}")

    # 9. 可视化
    plt.figure(figsize=(18, 18))
    plt.imshow(trajectory_map)
    plt.title(f"场景: {scene_name}\n道路类型: {road_type}", fontsize=16)

    # 添加图例
    legend_elements = []
    for i, vehicle_id in enumerate(trajectories.keys()):
        hue = i * 180 // len(trajectories)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
        role = trajectories[vehicle_id][0].get('role', 'unknown')
        legend_elements.append(plt.Line2D([0], [0], color=color / 255, lw=4, label=f"ID {vehicle_id} ({role})"))

    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.axis('off')

    plt_path = os.path.join(output_dir, f"{scene_name}_map_visualization.png")
    plt.savefig(plt_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"可视化地图已保存至: {plt_path}")

    print(f"场景 {scene_name} 处理完成\n")

def process_dataset(root_path, output_base_dir):
    """处理整个数据集"""
    # 遍历场景类型 (type1_subtype1_accident等)
    for scenario_type in os.listdir(root_path):
        type_path = os.path.join(root_path, scenario_type)
        if not os.path.isdir(type_path):
            continue

        # 创建输出目录
        output_dir = os.path.join(output_base_dir, scenario_type)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n处理场景类型: {scenario_type}")

        # 获取meta文件
        meta_dir = os.path.join(type_path, "meta")
        if not os.path.exists(meta_dir):
            print(f"警告: 未找到meta目录 {meta_dir}")
            continue

        # 处理每个场景
        meta_files = [f for f in os.listdir(meta_dir) if f.endswith(".txt")]
        for meta_file in tqdm(meta_files, desc="处理场景"):
            scene_name = os.path.splitext(meta_file)[0]
            try:
                process_scenario(
                    type_path,
                    scene_name,
                    output_dir,
                    highlight_frame=20  # 高亮显示第30帧
                )
            except Exception as e:
                print(f"处理场景 {scene_name} 错误: {e}")
                continue


# ==================== 4. 主执行入口 ====================
if __name__ == "__main__":
    # 单独处理一个场景的示例
    process_scenario(
        "./data/train/type1_subtype1_accident",
        "Town01_type001_subtype0001_scenario00004",#Town01_type001_subtype0001_scenario00004
        "/home/carla_user/processed_results",
        highlight_frame=20
    )

    # #处理整个数据集
    # process_dataset(
    #     "./data/train",
    #     "/home/carla_user/processed_results"
    # )