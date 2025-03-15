from top_down_simulator import Simulator
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Map():
    def __init__(self, map_size=(100, 100), resolution=0.1):
        """
        初始化TSDF地图
        
        参数:
            map_size: 地图大小 (width, height)
            resolution: 地图分辨率(m)
        """
        self.map_size = map_size
        self.resolution = resolution
        
        # 创建TSDF地图，每个体素包含SDF值和权重
        self.tsdf_values = np.zeros(map_size)  # SDF值
        self.tsdf_weights = np.zeros(map_size)  # 权重
        
        # 地图分辨率(sigma_map)设置为5个网格
        self.sigma_map = 1  # σ_map (地图分辨率)
        
        # 截断距离应该是地图分辨率的倍数
        self.scaling_factor = 20.0  # t (缩放因子)
        self.truncation_distance = self.scaling_factor * self.sigma_map  # δ = t * σ_map
        
        # 权重阈值，用于判断已知和未知空间
        self.tau_w = 0.02
        
        # 状态地图: 0-未知，1-已知
        self.state_map = np.zeros(map_size)
        
        # 可视化相关
        self.display_scale = 8  # 显示比例
        self.surface = None
        self.color_map = plt.cm.get_cmap('jet')
        
        # 创建专用于地图显示的画面
        self.map_surface = None
        
    def update_map(self, depth, angles, robot_x, robot_y, robot_theta):
        """
        使用深度数据更新TSDF地图
        
        参数:
            depth: 深度值数组
            angles: 对应的角度数组（弧度，范围0-2π）
            robot_x, robot_y: 机器人位置
            robot_theta: 机器人朝向（弧度，范围0-2π）
        """
        for i, (d, angle) in enumerate(zip(depth, angles)):
            if d <= 0:
                continue
                
            # 计算光线方向（保持角度在0-2π范围内）
            ray_angle = angle % (2 * np.pi)
            
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
                
                # 计算体素中心坐标
                voxel_center_x = px + 0.5
                voxel_center_y = py + 0.5
                
                # 计算体素中心到相机的向量
                voxel_to_camera_vector = np.array([voxel_center_x - robot_x, voxel_center_y - robot_y])
                voxel_to_camera_distance = np.linalg.norm(voxel_to_camera_vector)
                
            
                # 计算该点到表面的SDF值
                # 使用projection_distance而不是s
                sdf = self._compute_sdf(voxel_to_camera_distance, d)
                
                # 计算权重 (式2)，考虑深度值的平方倒数
                base_weight = self._compute_weight(sdf)
                depth_weight = 1.0 / (d * d) if d > 0 else 0  # 基于深度的权重
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
    
    def _compute_sdf(self, voxel_to_camera_distance, ray_depth):
        """
        计算采样点的SDF值 (式1)
        
        参数:
            voxel_to_camera_distance: 体素中心投影到光线上的点与相机的距离
            ray_depth: 光线测量的深度值
        
        返回:
            SDF值
        """
        # 实现式(1)
    
        sdf = min(self.truncation_distance, 
                  max(-self.truncation_distance, ray_depth - voxel_to_camera_distance))
        return sdf
    
    def _compute_weight(self, sdf):
        """
        计算TSDF权重基础部分 (式2)
        
        参数:
            sdf: 有符号距离场值
        
        返回:
            权重值
        """
        # 实现式(2)
        if -self.sigma_map < sdf <= self.truncation_distance:
            return 1
        elif -self.truncation_distance <= sdf <= -self.sigma_map:
            return (self.truncation_distance + sdf) / (self.truncation_distance - self.sigma_map)
        else:
            return 0.0
    
    def deintegrate_frame(self, depth, angles, robot_x, robot_y, robot_theta):
        """
        从地图中移除一帧的贡献 (式4)
        
        参数:
            depth: 深度值数组
            angles: 对应的角度数组（弧度，范围0-2π）
            robot_x, robot_y: 机器人位置
            robot_theta: 机器人朝向（弧度，范围0-2π）
        """
        for i, (d, angle) in enumerate(zip(depth, angles)):
            if d <= 0:
                continue
                
            # 计算光线方向
            ray_angle = angle % (2 * np.pi)
            
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
                
                # 计算体素中心坐标
                voxel_center_x = px + 0.5
                voxel_center_y = py + 0.5
                
                # 计算体素中心到相机的向量
                voxel_to_camera_vector = np.array([voxel_center_x - robot_x, voxel_center_y - robot_y])
                voxel_to_camera_distance = np.linalg.norm(voxel_to_camera_vector)
                
                # 计算光线方向的单位向量
                ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
                
                # 计算体素中心在光线上的投影点
                projection_distance = np.dot(voxel_to_camera_vector, ray_direction)
                
                # 计算投影点的坐标
                projection_point = np.array([robot_x, robot_y]) + projection_distance * ray_direction
                
                # 计算体素中心到投影点的距离
                voxel_to_projection_distance = np.linalg.norm(
                    np.array([voxel_center_x, voxel_center_y]) - projection_point
                )
                
                # 计算该点到表面的SDF值
                sdf = self._compute_sdf(projection_distance, d, voxel_to_projection_distance)
                
                # 计算权重，考虑深度值
                base_weight = self._compute_weight(sdf)
                depth_weight = 1.0 / (d * d) if d > 0 else 0
                weight = base_weight * depth_weight
                
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
                if abs(self.tsdf_values[x, y]) < 0.4 and self.tsdf_weights[x, y] > self.tau_w:
                    surface_points.append((x, y))
        
        return surface_points
    
    def create_map_image(self, robot_x, robot_y, robot_theta):
        """
        创建地图的新图像
        
        参数:
            robot_x, robot_y: 机器人位置
            robot_theta: 机器人朝向
        
        返回:
            地图的Pygame表面对象
        """
        # 创建matplotlib图形
        fig = plt.figure(figsize=(5, 5), dpi=80)
        ax = fig.add_subplot(111)
        
        # 绘制TSDF地图
        # 使用权重过滤未知区域
        masked_tsdf = np.ma.masked_where(self.tsdf_weights < self.tau_w, self.tsdf_values) # 滤掉不可靠的TSDF(截断有符号距离场)值
        
        # 裁剪TSDF值范围
        clipped_tsdf = np.clip(masked_tsdf, -self.truncation_distance, self.truncation_distance)
        
        # 正规化为0-1范围以用于着色
        normalized_tsdf = (clipped_tsdf + self.truncation_distance) / (2 * self.truncation_distance)
        
        # 绘制颜色映射的TSDF
        cax = ax.imshow(normalized_tsdf.T, cmap='seismic', origin='lower', 
                        extent=(0, self.map_size[0], 0, self.map_size[1]))
        
        # 绘制状态图的轮廓（已知区域）
        contour = ax.contour(np.arange(self.map_size[0]), np.arange(self.map_size[1]), 
                           self.state_map.T, colors='blue', linestyles='solid', levels=[0.5])
        
        # 获取表面点并绘制
        surface_points = self.get_surface_points()
        if surface_points:
            sp_x, sp_y = zip(*surface_points)
            ax.scatter(sp_x, sp_y, color='black', s=1, alpha=0.7)
        
        # 绘制机器人位置和朝向
        robot_marker_size = 10
        ax.plot(robot_x, robot_y, 'ro', markersize=robot_marker_size)
        # 绘制方向箭头
        arrow_length = 5
        ax.arrow(robot_x, robot_y, 
                 arrow_length * np.cos(robot_theta), 
                 arrow_length * np.sin(robot_theta),
                 head_width=2, head_length=2, fc='r', ec='r')
        
        ax.set_title('TSDF Map with Robot Position')
        fig.colorbar(cax, label='Normalized SDF')
        
        # 转换为Pygame表面
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        size = canvas.get_width_height()
        map_surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # 清理matplotlib图形
        plt.close(fig)
        
        return map_surf
    
    def render_map(self, screen, offset_x=700, offset_y=100, robot_x=0, robot_y=0, robot_theta=0):
        """
        在屏幕上渲染TSDF地图
        
        参数:
            screen: Pygame屏幕对象
            offset_x, offset_y: 绘制偏移量
            robot_x, robot_y: 机器人位置
            robot_theta: 机器人朝向
        """
        # 创建新的地图图像
        self.map_surface = self.create_map_image(robot_x, robot_y, robot_theta)
        
        # 在屏幕上显示图表
        screen.blit(self.map_surface, (offset_x, offset_y))

    def add_debug_info(self, screen, x, y, depth, angles):
        """
        添加调试信息到屏幕
        
        参数:
            screen: Pygame屏幕对象
            x, y: 机器人位置
            depth: 深度值数组
            angles: 角度数组
        """
        # 创建字体
        font = pygame.font.SysFont('Arial', 14)
        
        # 显示调试信息
        info_lines = [
            f"Robot Position: ({x:.1f}, {y:.1f})",
            f"Map Coverage: {np.sum(self.state_map > 0) / (self.map_size[0] * self.map_size[1]) * 100:.1f}%",
            f"Total Points: {len(depth)}",
            f"Angle Range: {np.degrees(min(angles)):.1f}° - {np.degrees(max(angles)):.1f}°"
        ]
        
        for i, line in enumerate(info_lines):
            text_surface = font.render(line, True, (0, 0, 0))
            screen.blit(text_surface, (700, 20 + i * 20))

if __name__ == "__main__":
    sim = Simulator(size=(100, 100), window_size=(1200, 800))
    
    mapping_module = Map(map_size=(100, 100), resolution=0.1)

    while sim.running:
        # 处理事件
        sim.handle_events()
        
        # 获取最新深度图 depth and corresponding angles both len=90
        depth, angles = sim.render_image()
        
        # robot_pose
        x, y, theta = sim.get_robot_pose()
        
        # 更新TSDF地图
        mapping_module.update_map(depth, angles, x, y, theta)
        
        # 清空屏幕
        sim.screen.fill((255, 255, 255))
        
        # 绘制模拟器
        sim.plot_sim()
        
        # 渲染TSDF地图（传入机器人位置和朝向）
        mapping_module.render_map(sim.screen, offset_x=700, offset_y=100, robot_x=x, robot_y=y, robot_theta=theta)
        
        # 添加调试信息
        mapping_module.add_debug_info(sim.screen, x, y, depth, angles)
        
        # 更新显示
        pygame.display.flip()
        
        # 控制帧率
        sim.clock.tick(30)