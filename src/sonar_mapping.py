from side_view_simulator import UnderwaterSimulator
import pygame, yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import random

class SonarMap:
    def __init__(self,robot_config_path,  map_size=(500, 100)):
        """
        初始化侧视图TSDF地图
        
        参数:
            map_size: 地图大小 (width, height)
            resolution: 地图分辨率(m)
        """
        with open(robot_config_path, 'r') as file:
            config = yaml.safe_load(file)
            sensor_fov = int(config['sensor_fov'])
            fov_partition_num = int(config['sensor_fov_partition_num'])
            self.sensor_range = int(config['sensor_range'])
            sensor_range_resolution = float(config['sensor_range_resolution'])
        
        # Assume we have multiple rays from sensors based on angle resolution
        self.rays_angles = np.linspace(-0.5*np.radians(sensor_fov), 0.5*np.radians(sensor_fov), fov_partition_num+1)
        self.range_axis = np.arange(0, self.sensor_range, sensor_range_resolution)
        
        self.map_size = map_size
                
        # 创建TSDF地图，每个体素包含SDF值和权重
        self.tsdf_values = np.zeros(map_size)  # SDF值
        self.tsdf_weights = np.zeros(map_size)  # 权重
        
        # 地图分辨率(sigma_map)设置为1个网格 BY DEFAULT
        self.sigma_map = 1  # σ_map (地图分辨率)
        
        # 截断距离应该是地图分辨率的倍数
        self.scaling_factor = 0.5*self.sensor_range / self.sigma_map  # t (缩放因子)
                                                                # t=0.5 is reasonable because 
                                                                # assume 60m sensing range, then truncatio_distance(td)=30,
                                                                # and a obstacle happen to locate at 60.0001m
                                                                # so not detected, ideally, from 0~30 start from cam, tsdf=30, 
                                                                # from 30~60 start from cam, tsdt=30~0, but with info we have we cannot know, 
                                                                # so best strategy is not to update this range 
                                                                # (i.e., the ray that update tsdf will stop update this range) 
                                                                # later max_range for updating tsdf will set to self.truncation_distance if the true range > sensing range (i.e., sensing nothing) 
        self.truncation_distance = self.scaling_factor * self.sigma_map  # δ = t * σ_map
        
        # 权重阈值，用于判断已知和未知空间
        self.tau_w = 5
        
        # 状态地图: 0-未知，1-已知
        self.state_map = np.zeros(map_size)
        
        # 可视化相关
        self.display_scale = 8  # 显示比例
        self.surface = None
        self.color_map = plt.cm.get_cmap('jet')
        
    def update_map(self, depth_map, angles, robot_x, robot_y, robot_pitch):
        """
        使用深度数据更新TSDF地图
        
        参数:
            depth_map: 深度值数组
            angles: 对应的角度数组
            robot_x, robot_y: 机器人位置
            robot_pitch: 机器人pitch角度（朝向）
        """
        for i, (d, angle) in enumerate(zip(depth_map, angles)):
                
            # 计算光线方向
            ray_angle = angle
            
            # 对光线上的点进行采样
            if d <= 0:
                max_range = self.truncation_distance
            else:
                max_range = min(d + self.truncation_distance, d * 2) # update voxel with (self.truncation_distance) behind the obstacle
            
            # 在光线方向上均匀采样点
            sample_points = np.linspace(0, max_range, int(max_range) + 1)
            
            for s in sample_points:
                # 计算采样点在地图上的坐标
                px = int(robot_x + s * np.cos(ray_angle))
                py = int(robot_y + s * np.sin(ray_angle))
                
                # 检查点是否在地图范围内
                if not (0 <= px < self.map_size[0] and 0 <= py < self.map_size[1]):
                    continue
                
                # 计算该点到表面的SDF值
                if d <= 0:
                    sdf = self.truncation_distance
                else:
                    sdf = self._compute_sdf(s, d)
                
                
                # 计算权重 (式2)
                base_weight = self._compute_weight(sdf)
                # depth_weight = 1.0 / (d * d) if d > 0 else 1  # 基于深度的权重
                
                # depth置信度 = 最大置信度 × e^(-k × 距离)
                    # 其中：

                    # 最大置信度：近距离探测的最高置信度(通常设为0.95或0.99)
                    # k：衰减系数(根据设备性能调整，典型值在0.05-0.2之间)
                    # 距离：以米为单位的探测距离
                    # 例如，如果设置最大置信度为0.95，k=0.1：

                    # 10米处置信度 = 0.95 × e^(-0.01×10) ≈ 0.86
                    # 100米处置信度 = 0.95 × e^(-0.01×100) ≈ 0.35
                depth_weight = 0.95*np.exp(-0.01 * d) if d > 0 else 0.95  # 基于深度的权重
                weight = base_weight * depth_weight
                
                # 更新TSDF值和权重 (式3)
                old_tsdf = self.tsdf_values[px, py]
                old_weight = self.tsdf_weights[px, py]
                
                # 更新公式(3)
                if old_weight + weight > 0:
                    self.tsdf_values[px, py] = (old_tsdf * old_weight + sdf * weight) / (old_weight + weight)
                    self.tsdf_weights[px, py] = old_weight + weight
                
                # 更新状态图 (式5)
                if 0.0 < self.tsdf_weights[px, py] <= self.tau_w:
                    self.state_map[px, py] = 0  # 未知
                else:
                    self.state_map[px, py] = 1  # 已知
            print(np.max(self.tsdf_weights))
            
    def update_sonar_map(self, robot_x, robot_y, depth2seafloar, robot_pitch, sonar_image):
        """
        使用深度数据更新TSDF地图
        
        参数:
            depth2seafloar: depth2seafloar
            angles: 对应的角度数组
            robot_x, robot_y: 机器人位置
            robot_pitch: 机器人pitch角度（朝向）
        """
        angles = self.rays_angles + robot_pitch
        
        def get_prior_kernel_matrix():
            prior_kernel_matrix = np.zeros(shape=(len(self.range_axis), len(angles)))
            # for c, theta in enumerate(angles):
            #     distance = np.abs(depth2seafloar/np.sin(theta))
            #     if distance < self.sensor_range:
            #         r_idx = np.abs(self.range_axis - distance).argmin()
            #         prior_kernel_matrix[r_idx][c] += 1
            
            
            distances = np.abs(depth2seafloar/np.sin(angles))
            valid_mask = distances < self.sensor_range
            valid_distances = distances[valid_mask]
            valid_angles_indices = np.where(valid_mask)[0]
            r_indices = np.abs(self.range_axis[:, np.newaxis] - valid_distances).argmin(axis=0)
            prior_kernel_matrix[r_indices, valid_angles_indices] += 1
            # np.savetxt('prior_kernel_matrix.txt', prior_kernel_matrix, fmt='%.0f', delimiter=',')
            
            # sonar_image = sonar_image.reshape(-1,1)
            # depth_image = range_axis.reshape(1,-1) @ prior_kernel_matrix
            return prior_kernel_matrix
        
        def reconstruct_depth_from_sonar_byprior(sonar_image):
            # the logic is expanding sonaring image column by column (really naive but effective)
            # [[0] => [[0,0,0],
            #  [2]     [0,1,1],
            #  [1]]    [1,0,0]]
                    
            depth_image = np.full(len(angles), -1.)
            
            values = []
            for count, value in zip(sonar_image, self.range_axis):
                values.extend([value] * count)
            
            # 排序并反转，确保从大到小
            values.sort(reverse=True)
            
            # 填充结果数组
            depth_image[len(angles)-len(values):] = values
            
            return depth_image
        
        depth_image_prior = reconstruct_depth_from_sonar_byprior(sonar_image)
        
        
        for i, (d, angle) in enumerate(zip(depth_image_prior, angles)):
                
            # 计算光线方向
            ray_angle = angle
            
            # 对光线上的点进行采样
            if d <= 0:
                max_range = self.truncation_distance
            else:
                max_range = min(d + self.truncation_distance, d * 2) # update voxel with (self.truncation_distance) behind the obstacle
            
            # 在光线方向上均匀采样点
            sample_points = np.linspace(0, max_range, int(max_range) + 1)
            
            for s in sample_points:
                # 计算采样点在地图上的坐标
                px = int(robot_x + s * np.cos(ray_angle))
                py = int(robot_y + s * np.sin(ray_angle))
                
                # 检查点是否在地图范围内
                if not (0 <= px < self.map_size[0] and 0 <= py < self.map_size[1]):
                    continue
                
                # 计算该点到表面的SDF值
                if d <= 0:
                    sdf = self.truncation_distance
                else:
                    sdf = self._compute_sdf(s, d)
                
                
                # 计算权重 (式2)
                base_weight = self._compute_weight(sdf)
                # depth_weight = 1.0 / (d * d) if d > 0 else 1  # 基于深度的权重
                
                # depth置信度 = 最大置信度 × e^(-k × 距离)
                    # 其中：

                    # 最大置信度：近距离探测的最高置信度(通常设为0.95或0.99)
                    # k：衰减系数(根据设备性能调整，典型值在0.05-0.2之间)
                    # 距离：以米为单位的探测距离
                    # 例如，如果设置最大置信度为0.95，k=0.1：

                    # 10米处置信度 = 0.95 × e^(-0.01×10) ≈ 0.86
                    # 100米处置信度 = 0.95 × e^(-0.01×100) ≈ 0.35
                depth_weight = 0.95*np.exp(-0.01 * d) if d > 0 else 0.95  # 基于深度的权重
                weight = base_weight * depth_weight
                
                # 更新TSDF值和权重 (式3)
                old_tsdf = self.tsdf_values[px, py]
                old_weight = self.tsdf_weights[px, py]
                
                # 更新公式(3)
                if old_weight + weight > 0:
                    self.tsdf_values[px, py] = (old_tsdf * old_weight + sdf * weight) / (old_weight + weight)
                    self.tsdf_weights[px, py] = old_weight + weight
                
                # 更新状态图 (式5)
                if 0.0 < self.tsdf_weights[px, py] <= self.tau_w:
                    self.state_map[px, py] = 0  # 未知
                else:
                    self.state_map[px, py] = 1  # 已知
            # print(np.max(self.tsdf_weights))
            
    def _compute_sdf(self, sample_dist, ray_depth):
        """
        计算采样点的SDF值 (式1)
        
        参数:
            sample_dist: 采样点到机器人的距离
            ray_depth: 光线测量的深度值
        
        返回:
            SDF值
        """
        # 实现式(1) - 计算有符号距离场
        sdf = min(self.truncation_distance, 
                  max(-self.truncation_distance, ray_depth - sample_dist))
        return sdf
    
    def _compute_weight(self, sdf):
        """
        计算TSDF权重 (式2)
        
        参数:
            sdf: 有符号距离场值
        
        返回:
            权重值
        """
        # 实现式(2)
        if 0 <= sdf <= self.truncation_distance:
            return 1.0
        elif -self.truncation_distance <= sdf < -self.sigma_map:
            # return (self.truncation_distance + sdf) / (self.truncation_distance - self.sigma_map)
            return np.exp(0.2*sdf)
        else:
            return 0.0
    
    def deintegrate_frame(self, depth, angles, robot_x, robot_y, robot_pitch):
        """
        从地图中移除一帧的贡献 (式4)
        
        参数:
            depth: 深度值数组
            angles: 对应的角度数组
            robot_x, robot_y: 机器人位置
            robot_pitch: 机器人朝向
        """
        for i, (d, angle) in enumerate(zip(depth, angles)):
            if d <= 0:
                continue
                
            # 计算光线方向
            ray_angle = angle
            
            # 对光线上的点进行采样
            max_range = min(d + self.truncation_distance, d * 2)
            
            # 在光线方向上均匀采样点
            sample_points = np.linspace(0, max_range, int(max_range) + 1)
            
            for s in sample_points:
                # 计算采样点在地图上的坐标
                px = int(robot_x + s * np.cos(ray_angle))
                py = int(robot_y + s * np.sin(ray_angle))
                
                # 检查点是否在地图范围内
                if not (0 <= px < self.map_size[0] and 0 <= py < self.map_size[1]):
                    continue
                
                # 计算该点到表面的SDF值
                sdf = self._compute_sdf(s, d)
                
                # 计算权重
                weight = self._compute_weight(sdf)
                
                # 更新TSDF值和权重 (式4)
                old_tsdf = self.tsdf_values[px, py]
                old_weight = self.tsdf_weights[px, py]
                
                if old_weight > weight:
                    self.tsdf_values[px, py] = (old_tsdf * old_weight - sdf * weight) / (old_weight - weight)
                    self.tsdf_weights[px, py] = old_weight - weight
    
    def get_surface_points(self):
        """
        从TSDF中提取表面点（零交叉点）
        
        返回:
            表面点的坐标数组
        """
        surface_points = []
        
        # 寻找TSDF值接近0的点
        for x in range(1, self.map_size[0]-1):
            for y in range(1, self.map_size[1]-1):
                if abs(self.tsdf_values[x, y]) < 2 and self.tsdf_weights[x, y] > 0:
                    surface_points.append((x, y))
        
        return surface_points
    
    def render_map(self, screen, offset_x=50, offset_y=500, robot_x=0, robot_y=0, robot_pitch=0):
        """
        在屏幕上渲染TSDF地图
        
        参数:
            screen: Pygame屏幕对象
            offset_x, offset_y: 绘制偏移量
            robot_x, robot_y: 机器人位置
            robot_pitch: 机器人pitch角度
        """
        # 创建matplotlib图形
        fig = plt.figure(figsize=(15, 4), dpi=80)
        ax = fig.add_subplot(111)
        
        # 绘制TSDF地图
        masked_tsdf = self.tsdf_values
        
        # 裁剪TSDF值范围
        clipped_tsdf = np.clip(masked_tsdf, -self.truncation_distance, self.truncation_distance)
        
        # 正规化为0-1范围以用于着色
        normalized_tsdf = (clipped_tsdf + self.truncation_distance) / (2 * self.truncation_distance)
        
        # 绘制颜色映射的TSDF
        cax = ax.imshow(normalized_tsdf.T, cmap='summer', origin='lower', 
                        extent=(0, self.map_size[0], 0, self.map_size[1]))
        
        # 绘制状态图的轮廓（已知区域）
        contour = ax.contour(np.arange(self.map_size[0]), np.arange(self.map_size[1]), 
                           self.state_map.T, colors='blue', linestyles='solid', levels=[0.5])
        
        # 获取表面点并绘制
        surface_points = self.get_surface_points()
        if surface_points:
            sp_x, sp_y = zip(*surface_points)
            ax.scatter(sp_x, sp_y, color='black', s=3, alpha=1)
        
        # 绘制机器人位置和朝向
        robot_marker_size = 10
        ax.plot(robot_x, robot_y, 'ro', markersize=robot_marker_size)
        
        # 绘制方向箭头 (在侧视图中，pitch角表示垂直平面中的方向)
        arrow_length = 10
        dx = arrow_length * np.cos(robot_pitch)  # 侧面视图中朝向向下为正方向
        dy = arrow_length * np.sin(robot_pitch)
        ax.arrow(robot_x, robot_y, dx, dy, head_width=6, head_length=6, fc='r', ec='r')
        
        ax.set_title('TSDF')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(cax, label='SDF', fraction=0.01)
        ax.invert_yaxis()
        
        # 转换为Pygame表面
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        size = canvas.get_width_height()
        map_surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # 在屏幕上显示图表
        screen.blit(map_surf, (offset_x, offset_y))
        
        # 清理matplotlib图形
        plt.close(fig)


if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)

    robot_config_path = '/home/clp/catkin_ws/src/sonar_map/src/config/robot_config.yaml'
    sim = UnderwaterSimulator(robot_config_path)
    
    # 创建侧视图地图实例
    mapping_module = SonarMap(robot_config_path)

    while sim.running:
        # 处理事件
        sim.handle_events()
        sim.plot_sim()
        
        # 获取最新深度图
        sonar_image, kernel_matrix, range_axis, depth_image, angles = sim.render_sonar()
        # 获取机器人位姿
        x, y, robot_pitch, depth = sim.get_robot_pose() # depth is here distance to Seafloor
        
        
        # 更新TSDF地图
        # mapping_module.update_map(depth_image, angles, robot_x, robot_y, robot_pitch)
        mapping_module.update_sonar_map(x, y, depth, robot_pitch, sonar_image)
        # 渲染TSDF地图
        
        mapping_module.render_map(sim.screen, robot_x=x, robot_y=y, robot_pitch=robot_pitch)
        
        # 绘制模拟器
        
        # 更新显示
        pygame.display.flip()
        
        # 控制帧率
        sim.clock.tick(50)