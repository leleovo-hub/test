import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2


class BEVMapExtractor:
    def __init__(self, class_mapping=None):
        """
        初始化BEV地图提取器

        参数:
            class_mapping: 可选，语义类ID到名称的映射字典
        """
        # 默认的Carla语义类映射
        self.default_class_mapping = {
            0: "Unlabeled", 1: "Building", 2: "Fence", 3: "Other", 4: "Pedestrian",
            5: "Pole", 6: "RoadLine", 7: "Road", 8: "SideWalk", 9: "Vegetation",
            10: "Vehicles", 11: "Wall", 12: "TrafficSign", 13: "Sky", 14: "Ground",
            15: "Bridge", 16: "RailTrack", 17: "GuardRail", 18: "TrafficLight",
            19: "Static", 20: "Dynamic", 21: "Water", 22: "Terrain"
        }

        # 使用用户提供的映射或默认映射
        self.class_mapping = class_mapping or self.default_class_mapping

        # 定义可视化颜色映射
        self.color_map = {
            0: [0, 0, 0], 1: [70, 70, 70], 2: [100, 40, 40], 3: [55, 90, 80],
            4: [220, 20, 60], 5: [153, 153, 153], 6: [157, 234, 50], 7: [128, 64, 128],
            8: [244, 35, 232], 9: [107, 142, 35], 10: [0, 0, 142], 11: [102, 102, 156],
            12: [220, 220, 0], 13: [70, 130, 180], 14: [81, 0, 81], 15: [150, 100, 100],
            16: [230, 150, 140], 17: [180, 165, 180], 18: [250, 170, 30], 19: [110, 190, 160],
            20: [170, 120, 50], 21: [45, 60, 150], 22: [145, 170, 100]
        }

    def load_bev_file(self, file_path):
        """
        加载BEV语义分割NPZ文件

        参数:
            file_path: NPZ文件路径

        返回:
            形状为(1200, 1200, 3)的NumPy数组
        """
        try:
            data = np.load(file_path)
            # 假设数组的键名为'data'，如果不同需要调整
            array_key = list(data.keys())[0] if len(data.keys()) == 1 else 'data'
            bev_array = data[array_key]

            # 验证数组形状
            if bev_array.shape != (1200, 1200, 3):
                print(f"警告: 数组形状不是(1200, 1200, 3)，而是{bev_array.shape}")

            return bev_array
        except Exception as e:
            print(f"加载文件失败: {e}")
            return None

    def get_vehicle_centers(self, class_map, instance_map, min_vehicle_area=50):
        """
        获取BEV地图中所有车辆的中心点像素坐标

        参数:
            class_map: 语义类别地图，形状为(1200, 1200)
            instance_map: 实例ID地图，形状为(1200, 1200)
            min_vehicle_area: 最小车辆面积阈值，小于此值的实例将被忽略

        返回:
            车辆中心点坐标列表，格式为[(x1, y1), (x2, y2), ...]
            以及对应的实例ID列表
        """
        # 获取车辆类别的掩码
        vehicle_mask = (class_map == 10).astype(np.uint8)  # 10是车辆的类别ID

        # 确保车辆掩码和实例地图结合
        vehicle_instances = instance_map * vehicle_mask

        # 获取唯一的车辆实例ID（排除0，即背景）
        unique_vehicle_instances = np.unique(vehicle_instances)
        unique_vehicle_instances = unique_vehicle_instances[unique_vehicle_instances > 0]

        centers = []
        instance_ids = []

        for instance_id in unique_vehicle_instances:
            # 创建当前实例的掩码
            instance_mask = (vehicle_instances == instance_id).astype(np.uint8)

            # 计算实例面积
            area = np.sum(instance_mask)
            if area < min_vehicle_area:
                continue  # 跳过面积过小的实例

            # 查找轮廓
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # 计算轮廓的矩
                moments = cv2.moments(contours[0])

                # 计算质心（中心点）
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    centers.append((cx, cy))
                    instance_ids.append(instance_id)

        return centers, instance_ids

    def visualize_vehicle_centers(self, class_map, instance_map, centers, save_path=None):
        """
        可视化车辆中心点在BEV地图上的位置

        参数:
            class_map: 语义类别地图
            instance_map: 实例ID地图
            centers: 车辆中心点坐标列表
            save_path: 保存路径，可选
        """
        # 首先可视化类别地图
        color_image = self.visualize_class_map(class_map, save_path=None)

        # 在图像上绘制车辆中心点
        for cx, cy in centers:
            cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)  # 蓝色圆点标记中心点

        plt.figure(figsize=(10, 10))
        plt.imshow(color_image)
        plt.title("Vehicle Centers on BEV Map")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"车辆中心点可视化已保存到 {save_path}")

        plt.show()
        return color_image

    def extract_class_map(self, bev_array):
        """
        提取语义类别地图

        参数:
            bev_array: BEV数组，形状为(1200, 1200, 3)

        返回:
            形状为(1200, 1200)的NumPy数组，包含语义类别ID
        """
        return bev_array[:, :, 2].astype(np.uint8)

    def extract_instance_map(self, bev_array):
        """
        提取实例ID地图

        参数:
            bev_array: BEV数组，形状为(1200, 1200, 3)

        返回:
            形状为(1200, 1200)的NumPy数组，包含实例ID
        """
        # 合并前两个通道的实例ID
        # 这里假设前两个通道以某种方式组合形成唯一实例ID
        # 例如，可能是高位和低位组合
        instance_id = (bev_array[:, :, 0].astype(np.int32) << 16) | bev_array[:, :, 1].astype(np.int32)
        return instance_id

    def visualize_class_map(self, class_map, save_path=None):
        """
        可视化语义类别地图

        参数:
            class_map: 语义类别地图，形状为(1200, 1200)
            save_path: 可选，保存可视化结果的路径
        """
        # 创建彩色图像
        h, w = class_map.shape
        color_image = np.zeros((h, w, 3), dtype=np.uint8)

        # 为每个类别分配颜色
        for class_id, color in self.color_map.items():
            mask = (class_map == class_id)
            color_image[mask] = color

        # 显示图像
        plt.figure(figsize=(10, 10))
        plt.imshow(color_image)
        plt.title("Semantic Class Map")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"语义类别地图已保存到 {save_path}")

        plt.show()

        return color_image

    def visualize_instance_map(self, instance_map, save_path=None):
        """
        可视化实例ID地图

        参数:
            instance_map: 实例ID地图，形状为(1200, 1200)
            save_path: 可选，保存可视化结果的路径
        """
        # 创建彩色图像
        h, w = instance_map.shape
        color_image = np.zeros((h, w, 3), dtype=np.uint8)

        # 获取唯一的实例ID
        unique_instances = np.unique(instance_map)

        # 为每个实例分配随机颜色
        np.random.seed(42)  # 固定随机种子以确保一致性
        for instance_id in unique_instances:
            if instance_id == 0:  # 通常0表示背景或未标记
                continue
            mask = (instance_map == instance_id)
            color = np.random.randint(0, 256, 3)
            color_image[mask] = color

        # 显示图像
        plt.figure(figsize=(10, 10))
        plt.imshow(color_image)
        plt.title("Instance ID Map")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"实例ID地图已保存到 {save_path}")

        plt.show()

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

    def visualize_selected_classes(self, class_map, instance_map, selected_class_ids, save_path=None,
                                   smooth=True, min_keep_area=1500):
        """
        可视化选定类别并标记车辆中心点

        参数:
            class_map: 语义类别地图
            instance_map: 实例ID地图
            selected_class_ids: 要选择的类别ID列表
            save_path: 保存路径，可选
            smooth: 是否应用平滑处理
            min_keep_area: 保留的最小区域面积
        """
        h, w = class_map.shape
        color_image = np.full((h, w, 3), 255, dtype=np.uint8)  # 白色背景

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

        # 获取车辆中心点
        vehicle_centers, _ = self.get_vehicle_centers(class_map, instance_map)

        # 在车辆中心点添加标记
        for cx, cy in vehicle_centers:
            # 绘制红色十字标记
            cv2.drawMarker(color_image, (cx, cy), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        # 5. 只在目标mask上着色
        for class_id in selected_class_ids:
            if class_id not in self.color_map:
                continue
            color = self.color_map[class_id]
            class_mask = (class_map == class_id) & (mask == 1)
            color_image[class_mask] = color

        plt.figure(figsize=(12, 12))
        plt.imshow(color_image, aspect='equal')
        plt.title("Selected Classes with Vehicle Centers")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        return color_image

    def process_directory(self, input_dir, output_dir=None, visualize=True):
        """
        处理目录中的所有BEV NPZ文件

        参数:
            input_dir: 输入目录路径
            output_dir: 输出目录路径，可选
            visualize: 是否可视化结果
        """
        # 创建输出目录（如果指定）
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取所有NPZ文件
        npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

        print(f"找到 {len(npz_files)} 个NPZ文件")

        for npz_file in tqdm(npz_files, desc="处理文件"):
            file_path = os.path.join(input_dir, npz_file)
            file_name = os.path.splitext(npz_file)[0]

            # 加载BEV数据
            bev_array = self.load_bev_file(file_path)
            if bev_array is None:
                continue

            # 提取类别和实例地图
            class_map = self.extract_class_map(bev_array)
            instance_map = self.extract_instance_map(bev_array)

            # 计算统计信息
            stats = self.calculate_statistics(class_map)

            print(f"\n文件: {npz_file}")
            print(f"地图覆盖率: {stats['coverage_percentage']:.2f}%")
            print("类别分布:")
            for class_name, percentage in sorted(stats['class_percentages'].items(),
                                                 key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {percentage:.2f}%")

            # 可视化
            if visualize:
                # 可视化类别地图
                if output_dir:
                    class_vis_path = os.path.join(output_dir, f"{file_name}_class_map.png")
                    self.visualize_class_map(class_map, class_vis_path)
                else:
                    self.visualize_class_map(class_map)

                # 可视化实例地图（只显示非背景实例）
                if np.any(instance_map > 0):
                    if output_dir:
                        instance_vis_path = os.path.join(output_dir, f"{file_name}_instance_map.png")
                        self.visualize_instance_map(instance_map, instance_vis_path)
                    else:
                        self.visualize_instance_map(instance_map)

                # 提取特定类别的区域示例（例如道路）
                road_mask = self.extract_specific_class(class_map, 7, min_area=100)
                if output_dir:
                    road_vis_path = os.path.join(output_dir, f"{file_name}_road_mask.png")
                    plt.figure(figsize=(10, 10))
                    plt.imshow(road_mask, cmap='gray')
                    plt.title("Road Mask")
                    plt.axis('off')
                    plt.savefig(road_vis_path, bbox_inches='tight')
                    plt.close()


# 使用示例
if __name__ == "__main__":
    # 创建提取器实例
    extractor = BEVMapExtractor()

    # 处理单个文件
    bev_file = r"C:\Users\HLC\Downloads\DeepAccident_val_01\type1_subtype1_accident\ego_vehicle\BEV_instance_camera\Town01_type001_subtype0001_scenario00004\Town01_type001_subtype0001_scenario00004_020.npz"
    bev_array = extractor.load_bev_file(bev_file)

    if bev_array is not None:
        # 提取语义类别地图和实例地图
        class_map = extractor.extract_class_map(bev_array)
        instance_map = extractor.extract_instance_map(bev_array)

        # 定义要保留的类别ID：6=RoadLine（车道线）、7=Road（道路）、10=Vehicles（车辆）
        selected_class_ids = [6, 7, 10]

        # 可视化选定类别并标记车辆位置
        extractor.visualize_selected_classes(
            class_map,
            instance_map,
            selected_class_ids,
            save_path="selected_classes_with_vehicle.png",
            smooth=True,
            min_keep_area=1500
        )

        # 提取特定类别的区域（例如道路）
        road_mask = extractor.extract_specific_class(class_map, 7)
        plt.figure(figsize=(10, 10))
        plt.imshow(road_mask, cmap='gray')
        plt.title("Road Mask")
        plt.axis('off')
        plt.show()

        # 计算统计信息
        stats = extractor.calculate_statistics(class_map)
        print("\n地图统计信息:")
        print(f"总类别数: {stats['total_classes']}")
        print(f"地图覆盖率: {stats['coverage_percentage']:.2f}%")

        # 获取车辆中心点
        vehicle_centers, instance_ids = extractor.get_vehicle_centers(class_map, instance_map)

        if vehicle_centers:
            print(f"找到 {len(vehicle_centers)} 辆车辆的中心点:")
            for i, (cx, cy) in enumerate(vehicle_centers):
                print(f"车辆 {i + 1} (实例ID: {instance_ids[i]}): 坐标 ({cx}, {cy})")

            # 可视化车辆中心点
            extractor.visualize_vehicle_centers(class_map, instance_map, vehicle_centers,
                                                save_path="vehicle_centers_visualization.png")
        else:
            print("未在BEV地图中找到车辆")

        # 获取车辆位置附近的类别信息（以第一个车辆为例）
        if vehicle_centers:
            cx, cy = vehicle_centers[0]  # 取第一个车辆的中心点
            vehicle_class = class_map[cy, cx]  # 注意y坐标在前
            class_name = extractor.class_mapping.get(vehicle_class, "Unknown")
            print(f"车辆1位置处的类别: {class_name} (ID: {vehicle_class})")

            # 获取车辆周围的实例信息
            vehicle_instance = instance_map[cy, cx]
            print(f"车辆1位置处的实例ID: {vehicle_instance}")