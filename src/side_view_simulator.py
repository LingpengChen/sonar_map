import numpy as np
import pygame
import sys, yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import random
from scipy.ndimage import gaussian_filter1d
import os


class UnderwaterSimulator:
    """
    基于 Pygame 的海底侧面视图二维环境模拟器
    模拟器提供更密集的网格，凹凸不平的海底地形，小石头和障碍物
    """
    
    def __init__(self, robot_config_path, size=(500, 100), window_size=(1200, 800)):
        """
        初始化模拟器环境和机器人
        
        参数:
            size: 环境网格尺寸 (宽度,高度)，默认500x100 (50m x 10m)
            window_size: Pygame窗口尺寸，默认1200x800
        """
        # 环境设置
        self.size = size
        # 创建环境地图 (1=自由空间/水, 0=障碍物/海底地形)
        self.environment = np.ones(size)
        
        # 机器人设置
        with open(robot_config_path, 'r') as file:
            config = yaml.safe_load(file)
            sensor_fov = int(config['sensor_fov'])
            sensor_fov_partition_num = int(config['sensor_fov_partition_num'])
            sensor_range = int(config['sensor_range'])
            sensor_range_resolution = float(config['sensor_range_resolution'])
        
        
        self.robot = {
            'x': size[0] // 20,     # 初始X位置（网格坐标）
            'y': size[1] // 2,     # 初始Y位置（网格坐标）
            'pitch': 0.08*np.pi,          # 初始水平方向（0=向右，π=向左，弧度）
            # 'pitch': 1.2*np.pi,          # 初始水平方向（0=向右，π=向左，弧度） 
            'radius': 1,            # 机器人半径
            'fov': np.radians(sensor_fov), # 视场角（设置为30度）
            'fov_partition_num': sensor_fov_partition_num, # 角度分辨率
            'max_range': sensor_range,       # 最大探测范围（网格数）
            'range_resolution': sensor_range_resolution,       
        }
        # range_axis [0, 0.5, 1, ..., 99.0, 99.5]
        self.range_axis = np.arange(0, self.robot['max_range'], self.robot['range_resolution'])
        
        
        # Assume we have multiple rays from sensors based on angle resolution
        start_angle = -0.5*self.robot['fov']
        end_angle = 0.5*self.robot['fov']
        line_num = self.robot['fov_partition_num']+1
        self.rays_angles = np.linspace(start_angle, end_angle, line_num)
        
        # 轨迹记录
        self.trajectory = [(self.robot['x'], self.robot['y'])]
        
        # Pygame设置
        self.window_size = window_size
        self.display_scale = min(window_size[0] / size[0], window_size[1] / size[1])
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None
        
        # 水粒子效果
        self.particles = []
        self.max_particles = 100
        
        # 控制参数
        self.running = True
        self.move_speed = 0.5      # 移动速度
        self.turn_speed = np.radians(5)  # 旋转速度
        self.show_rays = True      # 是否显示光线
        
        # 上次深度图
        self.last_depths = None
        self.last_angles = None
        
        # 颜色设置
        self.colors = {
            'water': (30, 100, 180),        # 深蓝色水
            'surface': (0, 150, 255),       # 浅蓝色水面
            'seafloor': (139, 115, 85),     # 棕色海底
            'rocks': (110, 110, 110),       # 灰色岩石
            'robot': (255, 50, 50),         # 红色机器人
            'ray': (255, 255, 0, 150),      # 黄色光线
            'trajectory': (0, 200, 255, 180) # 蓝色轨迹
        }
        
        self._init_pygame()
        
        print("海底环境模拟器已启动。使用WASD或箭头键移动，Q/E调整水平方向，Z/X调整pitch角度，R切换光线显示，ESC退出。")
        
    def _init_pygame(self):
        """初始化Pygame"""
        if not self.pygame_initialized:
            # 创建海底环境
            self.create_underwater_environment()
            
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Underwater Simulator")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 18)
            self.pygame_initialized = True
            
    
    def create_underwater_environment(self):
        """创建一个凹凸不平的海底环境，带有小石头和障碍物"""
        # 重置环境为全水域
        self.environment = np.ones(self.size)
        
        # 生成凹凸不平的海底基础地形
        seafloor_base = int(self.size[1] * 0.7)  # 海底基准线，位于高度70%处
        
        # 使用正弦函数创建起伏的海底
        x = np.arange(0, self.size[0])
        # 主要海底轮廓 - 大的起伏
        main_seafloor = seafloor_base + (np.sin(x * 0.03) * 8).astype(int) + (np.sin(x * 0.01) * 10).astype(int)
        
        # 添加随机的小起伏
        noise = np.random.randint(-2, 2, self.size[0])
        seafloor_height = np.maximum(main_seafloor + noise, seafloor_base - 15)
        seafloor_height = np.minimum(seafloor_height, self.size[1] - 5)  # 确保不超出底部
        
        # 设置海底地形
        for x in range(self.size[0]):
            height = seafloor_height[x]
            self.environment[x, height:] = 0  # 海底以下都是固体
        
       
        # 添加一些洞穴或突出物
        for _ in range(10):  # 添加5个较大的地形特征
            x = random.randint(50, self.size[0] - 51)
            
            # 找到海底高度
            y_seafloor = 0
            while y_seafloor < self.size[1] and self.environment[x, y_seafloor] == 1:
                y_seafloor += 1
            
            # 添加突起或凹陷
            # feature_type = random.choice(['overhang', 'cave', 'ridge'])
            
            # 悬崖突出物
            width = random.randint(10, 20)
            height = random.randint(5, 8)
            for dx in range(-width//2, width//2):
                for dy in range(height):
                    h_factor = 1 - abs(dx)/(width/2)  # 高度因子，中间高两边低
                    if random.random() < h_factor:
                        nx, ny = x + dx, y_seafloor - dy - 1
                        if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]:
                            self.environment[nx, ny] = 0
            
         
    def robot_move(self, dx=0, dy=0, dpitch=0):
        """
        移动机器人，更新位姿
        
        参数:
            dx: x方向移动（网格数）
            dy: y方向移动（网格数）
            dpitch: 水平旋转角度（弧度）
            
        返回:
            是否成功移动
        """
        # 计算新位置
        new_x = self.robot['x'] + dx
        new_y = self.robot['y'] + dy
        new_pitch = (self.robot['pitch'] + dpitch) % (2 * np.pi)
        
        # 检查新位置是否有效
        if self._is_valid_position(new_x, new_y):
            # 更新机器人位置
            self.robot['x'] = new_x
            self.robot['y'] = new_y
            self.robot['pitch'] = new_pitch
            
            # 记录轨迹
            self.trajectory.append((new_x, new_y))
            if len(self.trajectory) > 300:  # 限制轨迹长度
                self.trajectory.pop(0)
            return True
        
        return False
    
    def _is_valid_position(self, x, y):
        """
        检查位置是否有效（在地图范围内且不是障碍物）
        
        参数:
            x, y: 要检查的位置
            
        返回:
            位置是否有效
        """
        # 检查是否在地图范围内
        if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            return False
        
        # 检查机器人半径范围内的点
        radius = self.robot['radius']
        for r_x in range(max(0, int(x - radius)), min(self.size[0], int(x + radius + 1))):
            for r_y in range(max(0, int(y - radius)), min(self.size[1], int(y + radius + 1))):
                # 计算到中心的距离
                if (r_x - x)**2 + (r_y - y)**2 <= radius**2:
                    # 如果这个点是障碍物
                    if self.environment[r_x, r_y] == 0:
                        return False
        
        return True
    
    def render_image(self):
        """
        根据机器人位姿和环境渲染深度图
        
        返回:
            depths: 深度值数组
            angles: 对应的角度数组
        """
        # 确定基准角度（水平朝向）
        # center_angle = self.robot['pitch'] + np.pi if self.robot['pitch'] < np.pi else self.robot['pitch'] - np.pi
        center_angle = self.robot['pitch'] 
        
        # 计算视线方向范围
        # start_angle = center_angle - self.robot['fov'] / 2
        # end_angle = center_angle + self.robot['fov'] / 2
        # angles = np.arange(start_angle, end_angle, self.robot['fov_partition_num'])
        angles = center_angle + self.rays_angles
        # 初始化深度数组
        depths = np.zeros_like(angles)
        
        # print(np.rad2deg(angles))
        # 对每个角度进行光线投射
        for i, angle in enumerate(angles):
            # 使用光线投射获取深度
            depths[i] = self._cast_ray(self.robot['x'], self.robot['y'], angle)
        # 保存深度图数据
        self.last_depths = depths.copy()
        self.last_angles = angles
        depths[depths == self.robot['max_range']] = -1
        return depths, angles

    def render_sonar(self):
        '''
        先生成depth image后生成sonar image
        
        返回:
            arg2-5 is for 3D reconsturction, we will not use them for our mapping, but we will use them to train our 3D recover network
            arg1: sonar_image [0,1 (intensity value), 2, 0...], len = range/resolution, sonar image is summation of kernel_matrix along angles direction
            arg2: kernel_matrix [[0,1],[1,0]], shape = (range/resolution, fov/fov_resolution) fov here is angle of vertical aperture 
            arg3: range_axis [0, 0.5, 1, ..., 9.0, 9.5]
            arg4: depth_image, len = fov/fov_resolution, store depth informtion from top to down
            arg5: angles, the cooresponding angle for each depth data
            
            range_axis.reshape(1,-1) @ kernel_matrix = depth_image.reshape(1,-1)
            
            The goal of our neural network is to recover/infer the kernel_matrix from sonar_image 
            So that we can get depth_image
        '''
        depth_image, angles = self.render_image()
        
        
        kernel_matrix = (depth_image[:, np.newaxis] == self.range_axis).astype(int)
        sonar_image = np.sum(kernel_matrix, axis=0)
        
        kernel_matrix = kernel_matrix.T
        # np.savetxt('kernel_matrix.txt', kernel_matrix, fmt='%.0f', delimiter=',')
        return sonar_image, kernel_matrix, self.range_axis, depth_image, angles
    
    def get_robot_pose(self):
        # 找到海底高度
        y_seafloor = 0
        x = int(self.robot['x'])
        while y_seafloor < self.size[1] and self.environment[x, y_seafloor] == 1:
            y_seafloor += 1
        sea_floor_depth = y_seafloor-self.robot['y']
        return self.robot['x'], self.robot['y'], self.robot['pitch'], sea_floor_depth
    
    def _cast_ray(self, x, y, angle):
        """
        从给定位置沿给定角度投射光线，计算深度
        
        参数:
            x, y: 起始位置
            angle: 投射角度
            
        返回:
            深度值
        """
        # 使用精细的光线步长以提高精度
        step_size = self.robot['range_resolution']
        
        # 计算光线方向向量
        dx = np.cos(angle) * step_size
        dy = np.sin(angle) * step_size
        
        # 光线投射循环
        current_x, current_y = x, y
        for depth in range(1, int(self.robot['max_range'] / step_size) + 1):
            # 更新光线当前位置
            current_x += dx
            current_y += dy
            
            # 检查是否超出地图范围
            if not (0 <= int(current_x) < self.size[0] and 0 <= int(current_y) < self.size[1]):
                return depth * step_size
            
            # 检查是否碰到障碍物
            if self.environment[int(current_x), int(current_y)] == 0:
                return depth * step_size
        
        # 如果达到最大范围也没有碰到障碍物
        return self.robot['max_range']
    
    def _draw_environment(self):
        """在Pygame窗口中绘制海底环境"""
        # 绘制水背景
        water_gradient = np.linspace(0, 1, self.size[1])
        for y in range(self.size[1]):
            # 随深度渐变的水颜色
            depth_factor = water_gradient[y]
            water_color = (
                int(self.colors['surface'][0] * (1 - depth_factor) + self.colors['water'][0] * depth_factor),
                int(self.colors['surface'][1] * (1 - depth_factor) + self.colors['water'][1] * depth_factor),
                int(self.colors['surface'][2] * (1 - depth_factor) + self.colors['water'][2] * depth_factor)
            )
            pygame.draw.line(
                self.screen,
                water_color,
                (0, int(y * self.display_scale)),
                (int(self.size[0] * self.display_scale), int(y * self.display_scale))
            )
        
        # 绘制海底和岩石
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.environment[x, y] == 0:  # 障碍物/海底
                    # 在海底表面使用较浅的颜色
                    is_surface = False
                    if y > 0 and self.environment[x, y-1] == 1:
                        is_surface = True
                        
                    color = self.colors['rocks'] if is_surface else self.colors['seafloor']
                    
                    # 添加一些随机变化使海底看起来更自然
                    if is_surface:
                        r, g, b = color
                        noise = random.randint(-15, 15)
                        color = (
                            max(0, min(255, r + noise)),
                            max(0, min(255, g + noise)),
                            max(0, min(255, b + noise))
                        )
                    
                    rect = pygame.Rect(
                        int(x * self.display_scale),
                        int(y * self.display_scale),
                        int(self.display_scale + 1),  # 略微扩大以避免缝隙
                        int(self.display_scale + 1)
                    )
                    pygame.draw.rect(self.screen, color, rect)
        
    def _draw_robot(self):
        """在Pygame窗口中绘制机器人"""
        # 绘制轨迹
        # if len(self.trajectory) > 1:
        #     points = [(int(x * self.display_scale), 
        #                int(y * self.display_scale)) 
        #               for x, y in self.trajectory]
        #     pygame.draw.lines(self.screen, self.colors['trajectory'], False, points, 2)
        
        # 绘制机器人（圆形）
        robot_x = int(self.robot['x'] * self.display_scale)
        robot_y = int(self.robot['y'] * self.display_scale)
        robot_radius = int(self.robot['radius'] * self.display_scale)  # 略微缩小显示
        
        # 绘制机器人主体
        pygame.draw.circle(self.screen, self.colors['robot'], (robot_x, robot_y), robot_radius)
        
        # # 绘制机器人朝向（水平方向）
        # 绘制传感器光线
        if self.show_rays and self.last_depths is not None and self.last_angles is not None:
            for depth, angle in zip(self.last_depths, self.last_angles):
                end_x = robot_x + depth * self.display_scale * np.cos(angle)
                end_y = robot_y + depth * self.display_scale * np.sin(angle)
                pygame.draw.line(self.screen, self.colors['ray'], 
                                (robot_x, robot_y), (end_x, end_y), 1)

    def _draw_depth_chart(self):
        """在Pygame窗口右侧绘制深度图"""
        if self.last_depths is None or self.last_angles is None:
            return
        
        # 创建matplotlib图形
        fig = plt.figure(figsize=(5, 3), dpi=80)
        ax = fig.add_subplot(111)
        
        # 创建均匀分布在FOV范围内的角度点
        fov_start = np.degrees(-self.robot['fov']/2)
        fov_end = np.degrees(self.robot['fov']/2)
        num_points = len(self.last_depths)
        display_angles = np.linspace(fov_start, fov_end, num_points)
        
        # 绘制深度图
        ax.bar(display_angles, self.last_depths, width=(fov_end-fov_start)/(num_points-1), 
               align='center', alpha=0.7, color='aqua')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Depth')
        ax.set_title('Depth Image')
        ax.set_xlim([fov_start, fov_end])
        ax.set_ylim([0, self.robot['max_range']])
        ax.grid(True, alpha=0.3)
        
        # 设置背景色
        ax.set_facecolor((0.9, 0.9, 1.0))
        
        # 转换为Pygame表面
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        size = canvas.get_width_height()
        chart_surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # 在屏幕上显示图表
        chart_x = 20
        chart_y = 250
        self.screen.blit(chart_surf, (chart_x, chart_y))
        
        # 清理matplotlib图形
        plt.close(fig)
    
    def _draw_information(self):
        """绘制信息面板"""
        # 创建信息面板背景
        panel_width = 250
        panel_height = 160
        panel_x = self.window_size[0] - panel_width - 20
        panel_y = 250
        
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))  # 半透明黑色
        self.screen.blit(panel, (panel_x, panel_y))
        
        # 显示位置和朝向信息
        info_texts = [
            f"position: ({self.robot['x']:.1f}, {self.robot['y']:.1f})",
            f"pitch: {np.degrees(self.robot['pitch']):.1f}°",
            f"FOV: {np.degrees(self.robot['fov']):.1f}°",
            f"map size: {self.size[0]}x{self.size[1]} ({self.size[0]/10}m x {self.size[1]/10}m)"
        ]
        
        for i, text in enumerate(info_texts):
            text_surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surf, (panel_x + 10, panel_y + 10 + i * 25))
    
    def _draw_instructions(self):
        """在屏幕上显示控制说明"""
        instructions = [
            "W/↑: Up",
            "S/↓: down",
            "A/←: left",
            "D/→: right",
            "R: show fov",
            "ESC: quit"
        ]
        
        # 创建说明面板背景
        panel_width = 150
        panel_height = len(instructions) * 25 + 10
        panel_x = 500
        panel_y = self.window_size[1] - panel_height - 320
        
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))  # 半透明黑色
        self.screen.blit(panel, (panel_x, panel_y))
        
        # 显示说明文字
        for i, instruction in enumerate(instructions):
            text_surf = self.font.render(instruction, True, (255, 255, 255))
            self.screen.blit(text_surf, (panel_x + 10, panel_y + 10 + i * 25))
    
    def handle_events(self):
        """处理Pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.show_rays = not self.show_rays
        
        # 处理连续按键
        keys = pygame.key.get_pressed()
        
        # 移动控制
        dx, dy = 0, 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -self.move_speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = self.move_speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy = -self.move_speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy = self.move_speed
        
        # Pitch控制
        dpitch = 0
        if keys[pygame.K_q]:
            dpitch = self.turn_speed  # 逆时针旋转
        if keys[pygame.K_e]:
            dpitch = -self.turn_speed  # 顺时针旋转

        # 应用移动
        if dx != 0 or dy != 0 or dpitch != 0:
            self.robot_move(dx=dx, dy=dy, dpitch=dpitch)
    
    def plot_sim(self):
        """绘制整个模拟场景"""
        
        # 绘制场景
        self._draw_environment()
        self._draw_robot()
        self._draw_depth_chart()
        self._draw_information()
        self._draw_instructions()
    
    def export_depth_image(self, filename="depth_image.png"):
        """将当前深度图导出为图像文件"""
        if self.last_depths is None or self.last_angles is None:
            self.render_image()
        
        plt.figure(figsize=(10, 5))
        
        # 绘制条形图
        plt.subplot(1, 2, 1)
        angle_degrees = np.degrees(np.linspace(-self.robot['fov']/2, self.robot['fov']/2, len(self.last_depths)))
        plt.bar(angle_degrees, self.last_depths, width=np.degrees(self.robot['fov']/self.robot['fov_partition_num']), align='center', color='aqua')
        plt.xlabel('角度 (度)')
        plt.ylabel('深度')
        plt.title('深度图')
        plt.grid(True, alpha=0.3)
        
        # 绘制极坐标图
        plt.subplot(1, 2, 2, projection='polar')
        plt.polar(self.last_angles, self.last_depths, 'c-')
        plt.title('极坐标深度图')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"深度图已保存至 {filename}")
    
    def export_environment(self, filename="underwater_environment.png"):
        """将当前环境导出为图像文件"""
        plt.figure(figsize=(12, 8))
        
        # 创建水深度渐变的背景
        water_img = np.ones((self.size[1], self.size[0], 3))
        for y in range(self.size[1]):
            depth_factor = y / self.size[1]
            water_color = [
                (1 - depth_factor) * 0.7 + depth_factor * 0.1,  # R
                (1 - depth_factor) * 0.9 + depth_factor * 0.4,  # G
                (1 - depth_factor) * 1.0 + depth_factor * 0.7   # B
            ]
            water_img[y, :, :] = water_color
        
        # 将障碍物区域设置为棕色
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.environment[x, y] == 0:
                    # 设置为棕色（海底/岩石）
                    if y > 0 and self.environment[x, y-1] == 1:
                        # 表面更亮
                        water_img[y, x] = [0.6, 0.45, 0.3]
                    else:
                        water_img[y, x] = [0.5, 0.35, 0.2]
        
        plt.imshow(water_img, origin='upper')
        
        # 绘制机器人位置和朝向
        plt.plot(self.robot['x'], self.robot['y'], 'ro', markersize=10)
        
        # 绘制视线方向
        dir_x = self.robot['x'] + 15 * np.cos(self.robot['pitch'] + np.pi )
        dir_y = self.robot['y'] + 15 * np.sin(self.robot['pitch'] + np.pi )
        plt.plot([self.robot['x'], dir_x], [self.robot['y'], dir_y], 'r-')
        
        # 绘制轨迹
        traj_x = [pos[0] for pos in self.trajectory]
        traj_y = [pos[1] for pos in self.trajectory]
        plt.plot(traj_x, traj_y, 'c--', alpha=0.7)
        
        plt.title('海底环境地图')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"环境地图已保存至 {filename}")



# 示例：如何使用UnderwaterSimulator类
if __name__ == "__main__":
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)
    # 创建模拟器实例
    robot_config_path = '/home/clp/catkin_ws/src/sonar_map/src/config/robot_config.yaml'
    sim = UnderwaterSimulator(robot_config_path)
    
    stored_data_path = '/home/clp/catkin_ws/src/sonar_map/neural_network/data/raw_data'
    stored_kernel_matrices = []
    stored_prior_matrices = []
    stored_sonar_images = []
    previous_sonar_image = None
    
    
    while sim.running:
        # 处理事件
        sim.handle_events()
        
        # 获取最新深度图
        # depth, angle = sim.render_image()
        sonar_image, kernel_matrix, range_axis, depth_image, angles = sim.render_sonar()
        # 获取机器人位姿
        x, y, pitch, depth2seafloor = sim.get_robot_pose()

        
        def get_prior_kernel_matrix():
            prior_kernel_matrix = np.zeros(shape=kernel_matrix.shape)

            distances = np.abs(depth2seafloor/np.sin(angles))
            valid_mask = distances < sim.robot['max_range']
            valid_distances = distances[valid_mask]
            valid_angles_indices = np.where(valid_mask)[0]
            r_indices = np.abs(range_axis[:, np.newaxis] - valid_distances).argmin(axis=0)
            prior_kernel_matrix[r_indices, valid_angles_indices] += 1
            
            # sonar_image = sonar_image.reshape(-1,1)
            # depth_image = range_axis.reshape(1,-1) @ prior_kernel_matrix
            return prior_kernel_matrix
        
        prior_kernel_matrix = get_prior_kernel_matrix()
        
        
        # 绘制场景
        sim.plot_sim()
        
        # 更新显示
        pygame.display.flip()
        
        # 控制帧率
        sim.clock.tick(30)
        
        # np.savetxt('kernel_matrix.txt', kernel_matrix, fmt='%.0f', delimiter=',')
        # np.savetxt('prior_kernel_matrix.txt', prior_kernel_matrix, fmt='%.0f', delimiter=',')
        # np.savetxt('sonar_image.txt', sonar_image, fmt='%.0f', delimiter=',')
        if previous_sonar_image is None or not np.array_equal(sonar_image, previous_sonar_image):
            # 将新数据添加到存储列表
            stored_kernel_matrices.append(kernel_matrix)
            stored_prior_matrices.append(prior_kernel_matrix)
            stored_sonar_images.append(sonar_image)    
            previous_sonar_image = sonar_image.copy()
            
            data_num = len(stored_sonar_images)
            if data_num % 50 == 49:
                np.save(os.path.join(stored_data_path, f'true_matrices_{random_seed}.npy'), np.array(stored_kernel_matrices))
                np.save(os.path.join(stored_data_path,f'prior_matrices_{random_seed}.npy'), np.array(stored_prior_matrices))
                np.save(os.path.join(stored_data_path,f'obs_vectors_{random_seed}.npy'), np.array(stored_sonar_images))
                print(f"save {data_num}")