from side_view_simulator import UnderwaterSimulator
import pygame, yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class SideViewMap:
    def __init__(self,robot_config_path,  map_size=(500, 100)):
        """
        初始化侧视图TSDF地图
        
        参数:
            map_size: 地图大小 (width, height)
            resolution: 地图分辨率(m)
        """
        with open(robot_config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.sensor_fov = float(config['sensor_fov'])
            self.sensor_fov_resolution = float(config['sensor_fov_resolution'])
            self.sensor_range = float(config['sensor_range'])
            
        self.map_size = map_size
                
        # 创建TSDF地图，每个体素包含SDF值和权重
        self.tsdf_values = np.zeros(map_size)  # SDF值
        self.tsdf_weights = np.zeros(map_size)  # 权重
        
        # 地图分辨率(sigma_map)设置为1个网格 BY DEFAULT
        self.sigma_map = 1  # σ_map (地图分辨率)
        
        # 截断距离应该是地图分辨率的倍数
        self.scaling_factor = self.sensor_range / self.sigma_map  # t (缩放因子)
        self.truncation_distance = self.scaling_factor * self.sigma_map  # δ = t * σ_map
        
        # 权重阈值，用于判断已知和未知空间
        self.tau_w = 10
        
        # 状态地图: 0-未知，1-已知
        self.state_map = np.zeros(map_size)
        
        # 可视化相关
        self.display_scale = 8  # 显示比例
        self.surface = None
        self.color_map = plt.cm.get_cmap('jet')
        
    def update_map(self, depth, angles, robot_x, robot_y, robot_pitch):
        """
        使用深度数据更新TSDF地图
        
        参数:
            depth: 深度值数组
            angles: 对应的角度数组
            robot_x, robot_y: 机器人位置
            robot_pitch: 机器人pitch角度（朝向）
        """
        for i, (d, angle) in enumerate(zip(depth, angles)):
                
            # 计算光线方向
            ray_angle = angle
            
            # 对光线上的点进行采样
            if d <= 0:
                max_range = self.truncation_distance
            else:
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
                if d <= 0:
                    sdf = self.truncation_distance
                else:
                    sdf = self._compute_sdf(s, d)
                
                
                # 计算权重 (式2)
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
        if -self.sigma_map < sdf <= self.truncation_distance:
            return 1.0
        elif -self.truncation_distance <= sdf <= -self.sigma_map:
            return (self.truncation_distance + sdf) / (self.truncation_distance - self.sigma_map)
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

# 示例：如何使用SideViewMap类与侧视图模拟器配合

def depth_image2sonar(depth2D, sonar1D):
    pass

if __name__ == "__main__":
    # 创建侧视图模拟器实例
    robot_config_path = '/home/clp/catkin_ws/src/sonar_map/src/config/robot_config.yaml'
    sim = UnderwaterSimulator(robot_config_path)
    
    # 创建侧视图地图实例
    mapping_module = SideViewMap(robot_config_path)

    while sim.running:
        # 处理事件
        sim.handle_events()
        
        # 获取最新深度图
        depth, angles = sim.render_image()
        print(len(depth))
        # 获取机器人位姿
        x, y, pitch, height = sim.get_robot_pose()
        
        # 更新TSDF地图
        mapping_module.update_map(depth, angles, x, y, pitch)
        
        # 绘制模拟器
        sim.plot_sim()
        
        # 渲染TSDF地图
        mapping_module.render_map(sim.screen, robot_x=x, robot_y=y, robot_pitch=pitch)
        
        # 更新显示
        pygame.display.flip()
        
        # 控制帧率
        sim.clock.tick(50)