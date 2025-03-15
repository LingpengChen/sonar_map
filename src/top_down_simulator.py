import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Simulator:
    """
    基于 Pygame 的二维环境模拟器，包含环境和机器人，可以模拟深度相机获取深度图
    """
    
    def __init__(self, size=(100, 100), resolution=0.1, window_size=(1200, 800)):
        """
        初始化模拟器环境和机器人
        
        参数:
            size: 环境网格尺寸，默认100x100
            resolution: 每个网格的实际尺寸（米），默认0.1米
            window_size: Pygame窗口尺寸，默认1200x800
        """
        # 环境设置
        self.size = size
        self.resolution = resolution
        # 创建环境地图 (1=自由空间, 0=障碍物)
        self.environment = np.ones(size)
        
        # 机器人设置
        self.robot = {
            'x': size[0] // 2,  # 初始X位置（网格坐标）
            'y': size[1] // 2,  # 初始Y位置（网格坐标）
            'theta': 0.0,       # 初始方向（弧度）
            'fov': np.radians(90),  # 视场角（默认90度）
            'max_range': 50,    # 最大探测范围（网格数）
            'resolution': np.radians(1)  # 角度分辨率（默认1度）
        }
        
        # 轨迹记录
        self.trajectory = [(self.robot['x'], self.robot['y'])]
        
        # Pygame设置
        self.window_size = window_size
        self.display_scale = min(window_size[0] // (size[0] + 200), window_size[1] // size[1])
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None
        
        # 控制参数
        self.running = True
        self.move_speed = 1.0  # 移动速度
        self.turn_speed = np.radians(5)  # 旋转速度
        self.show_rays = True  # 是否显示光线
        
        # 上次深度图
        self.last_depths = None
        self.last_angles = None

        
        self._init_pygame()
        
        print("模拟器已启动。使用WASD或箭头键控制机器人，R切换光线显示，ESC退出。")
        
    def _init_pygame(self):
        """初始化Pygame"""
        if not self.pygame_initialized:
            
            self.create_room_environment(wall_thickness=2, room_margin=10)
            self.add_rectangle_obstacle(40, 40, 60, 60)  # 中央方形
            self.add_circular_obstacle(30, 70, 8)        # 左上角圆形
            
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("2D Simulator")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 18)
            self.pygame_initialized = True
            
            self.robot_move(dx=-20, dy=-20)
            
    
    def create_room_environment(self, wall_thickness=2, room_margin=10):
        """
        创建一个带墙壁的房间环境
        
        参数:
            wall_thickness: 墙壁厚度（网格数）
            room_margin: 房间与边界的距离（网格数）
        """
        # 重置环境为全自由空间
        self.environment = np.ones(self.size)
        
        # 添加外墙
        x_max, y_max = self.size
        margin = room_margin
        thickness = wall_thickness
        
        # 左墙
        self.environment[margin:x_max-margin, margin:margin+thickness] = 0
        # 右墙
        self.environment[margin:x_max-margin, y_max-margin-thickness:y_max-margin] = 0
        # 下墙
        self.environment[margin:margin+thickness, margin:y_max-margin] = 0
        # 上墙
        self.environment[x_max-margin-thickness:x_max-margin, margin:y_max-margin] = 0
    
    def add_rectangle_obstacle(self, x_min, y_min, x_max, y_max):
        """
        添加矩形障碍物
        
        参数:
            x_min, y_min: 障碍物左下角坐标
            x_max, y_max: 障碍物右上角坐标
        """
        self.environment[x_min:x_max, y_min:y_max] = 0
    
    def add_circular_obstacle(self, center_x, center_y, radius):
        """
        添加圆形障碍物
        
        参数:
            center_x, center_y: 圆心坐标
            radius: 半径（网格数）
        """
        xx, yy = np.mgrid[:self.size[0], :self.size[1]]
        circle = (xx - center_x)**2 + (yy - center_y)**2 <= radius**2
        self.environment[circle] = 0
    
    def robot_move(self, dx=0, dy=0, dtheta=0):
        """
        移动机器人，更新位姿
        
        参数:
            dx: x方向移动（网格数）
            dy: y方向移动（网格数）
            dtheta: 旋转角度（弧度）
            
        返回:
            是否成功移动
        """
        # 计算新位置
        new_x = self.robot['x'] + dx
        new_y = self.robot['y'] + dy
        new_theta = (self.robot['theta'] + dtheta) % (2 * np.pi)
        
        # 检查新位置是否有效
        if self._is_valid_position(new_x, new_y):
            # 更新机器人位置
            self.robot['x'] = new_x
            self.robot['y'] = new_y
            self.robot['theta'] = new_theta
            
            # 记录轨迹
            self.trajectory.append((new_x, new_y))
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
        x_int, y_int = int(x), int(y)
        if not (0 <= x_int < self.size[0] and 0 <= y_int < self.size[1]):
            return False
        
        # 检查是否是自由空间
        return self.environment[x_int, y_int] == 1
    
    def render_image(self):
        """
        根据机器人位姿和环境渲染深度图
        
        返回:
            depths: 深度值数组
            angles: 对应的角度数组
        """
        # 计算视线方向范围
        start_angle = self.robot['theta'] - self.robot['fov'] / 2
        end_angle = self.robot['theta'] + self.robot['fov'] / 2
        angles = np.arange(start_angle, end_angle, self.robot['resolution']) % (2 * np.pi)
        
        # 初始化深度数组
        depths = np.zeros_like(angles)
        
        # 对每个角度进行光线投射
        for i, angle in enumerate(angles):
            # 使用光线投射获取深度
            depths[i] = self._cast_ray(self.robot['x'], self.robot['y'], angle)
        
        # 保存深度图数据
        self.last_depths = depths
        self.last_angles = angles
        # print(np.min(depths))
        return depths, angles
    
    def get_robot_pose(self):
        return self.robot['x'], self.robot['y'], self.robot['theta']
    
    def _cast_ray(self, x, y, angle):
        """
        从给定位置沿给定角度投射光线，计算深度
        
        参数:
            x, y: 起始位置
            angle: 投射角度
            
        返回:
            深度值
        """
        # 计算光线方向向量
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # 光线投射循环
        for depth in range(1, self.robot['max_range'] + 1):
            # 计算光线当前位置
            current_x = int(x + depth * dx)
            current_y = int(y + depth * dy)
            
            # 检查是否超出地图范围
            if not (0 <= current_x < self.size[0] and 0 <= current_y < self.size[1]):
                return depth
            
            # 检查是否碰到障碍物
            if self.environment[current_x, current_y] == 0:
                return depth
        
        # 如果达到最大范围也没有碰到障碍物
        return self.robot['max_range']
    
    def _draw_environment(self):
        """在Pygame窗口中绘制环境"""
        # 绘制背景
        self.screen.fill((240, 240, 240))
        
        # 绘制环境网格
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                color = (255, 255, 255) if self.environment[x, y] == 1 else (0, 0, 0)
                rect = pygame.Rect(
                    y * self.display_scale,
                    x * self.display_scale,
                    self.display_scale,
                    self.display_scale
                )
                pygame.draw.rect(self.screen, color, rect)
                
        # 绘制网格线
        grid_color = (200, 200, 200)
        for x in range(0, self.size[0] + 1, 10):
            pygame.draw.line(
                self.screen,
                grid_color,
                (0, x * self.display_scale),
                (self.size[1] * self.display_scale, x * self.display_scale)
            )
        for y in range(0, self.size[1] + 1, 10):
            pygame.draw.line(
                self.screen,
                grid_color,
                (y * self.display_scale, 0),
                (y * self.display_scale, self.size[0] * self.display_scale)
            )
    
    def _draw_robot(self):
        """在Pygame窗口中绘制机器人"""
        # 绘制轨迹
        if len(self.trajectory) > 1:
            points = [(int(y * self.display_scale + self.display_scale/2), 
                       int(x * self.display_scale + self.display_scale/2)) 
                      for x, y in self.trajectory]
            pygame.draw.lines(self.screen, (0, 0, 255), False, points, 2)
        
        # 绘制机器人位置
        robot_x = int(self.robot['y'] * self.display_scale + self.display_scale/2)
        robot_y = int(self.robot['x'] * self.display_scale + self.display_scale/2)
        pygame.draw.circle(self.screen, (255, 0, 0), (robot_x, robot_y), 6)
        
        # 绘制机器人朝向
        direction_len = 15
        dir_x = robot_x + direction_len * np.sin(self.robot['theta'])
        dir_y = robot_y + direction_len * np.cos(self.robot['theta'])
        pygame.draw.line(self.screen, (255, 0, 0), (robot_x, robot_y), (dir_x, dir_y), 2)
        
        
        # 绘制传感器光线
        if self.show_rays and self.last_depths is not None and self.last_angles is not None:
            for depth, angle in zip(self.last_depths, self.last_angles):
                end_x = robot_x + depth * self.display_scale * np.sin(angle)
                end_y = robot_y + depth * self.display_scale * np.cos(angle)
                pygame.draw.line(self.screen, (255, 255, 0, 128), (robot_x, robot_y), (end_x, end_y), 1)
        
        # 绘制视场角扇形
        if self.robot['fov'] < 2*np.pi:  # 如果不是全方位视图
            # 绘制视场扇形
            angle_start = self.robot['theta'] - np.pi/2 - self.robot['fov']/2
            pygame.draw.arc(
                self.screen,
                (255, 0, 0, 128),
                pygame.Rect(robot_x - 20, robot_y - 20, 40, 40),
                angle_start,
                angle_start + self.robot['fov'],
                2
            )
            
        # 显示位置和方向信息
        pos_text = self.font.render(f"Position: ({self.robot['x']:.1f}, {self.robot['y']:.1f})", True, (0, 0, 0))
        angle_text = self.font.render(f"Orientation: {np.degrees(self.robot['theta']):.1f}°", True, (0, 0, 0))
        self.screen.blit(pos_text, (self.size[1] * self.display_scale + 20, 20))
        self.screen.blit(angle_text, (self.size[1] * self.display_scale + 20, 50))
        
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
        # print(len(display_angles), len(self.last_depths))
        # print(display_angles)
        ax.bar(display_angles, self.last_depths, width=(fov_end-fov_start)/(num_points-1), align='center', alpha=0.7)
        ax.set_xlabel('angle(degree)')
        ax.set_ylabel('depth')
        ax.set_title('depth image')
        ax.set_xlim([fov_start, fov_end])
        ax.set_ylim([0, self.robot['max_range']])
        
        # 转换为Pygame表面
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        size = canvas.get_width_height()
        chart_surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # 在屏幕上显示图表
        chart_x = self.size[1] * self.display_scale + 20
        chart_y = 100
        self.screen.blit(chart_surf, (chart_x, chart_y))
        
        # 清理matplotlib图形
        plt.close(fig)
    
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
        
        # 使用WASD和箭头键控制机器人
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            # 沿机器人朝向移动
            dx = self.move_speed * np.cos(self.robot['theta'])
            dy = self.move_speed * np.sin(self.robot['theta'])
            self.robot_move(dx=dx, dy=dy)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            # 沿机器人朝向的反方向移动
            dx = -self.move_speed * np.cos(self.robot['theta'])
            dy = -self.move_speed * np.sin(self.robot['theta'])
            self.robot_move(dx=dx, dy=dy)
        if keys[pygame.K_LEFT] or keys[pygame.K_q]:
            # 左转
            self.robot_move(dtheta=self.turn_speed)
        if keys[pygame.K_RIGHT] or keys[pygame.K_e]:
            # 右转
            self.robot_move(dtheta=-self.turn_speed)
        # 侧移
        if keys[pygame.K_d]:
            # 左移
            dx = self.move_speed * np.cos(self.robot['theta'] - np.pi/2)
            dy = self.move_speed * np.sin(self.robot['theta'] - np.pi/2)
            self.robot_move(dx=dx, dy=dy)
        if keys[pygame.K_a]:
            # 右移
            dx = self.move_speed * np.cos(self.robot['theta'] + np.pi/2)
            dy = self.move_speed * np.sin(self.robot['theta'] + np.pi/2)
            self.robot_move(dx=dx, dy=dy)
    
    def _draw_instructions(self):
        """在屏幕上显示控制说明"""
        instructions = [
            "控制说明:",
            "W/↑: forward",
            "S/↓: backward",
            "Q/←: turn left",
            "E/→: turn right",
            "A: left",
            "D: right",
            "R: show light",
            "ESC: quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.font.render(instruction, True, (0, 0, 0))
            y_pos = self.window_size[1] - 30 * (len(instructions) - i)
            self.screen.blit(text, (self.size[1] * self.display_scale + 20, y_pos))
    
    def plot_sim(self):
         # 绘制场景
        self._draw_environment()
        self._draw_robot()
        self._draw_depth_chart()
        self._draw_instructions()
    
    def get_depth_data(self):
        """获取当前深度数据，供其他模块使用"""
        if self.last_depths is None or self.last_angles is None:
            self.render_image()
        return self.last_depths, self.last_angles
    
    def export_depth_image(self, filename="depth_image.png"):
        """将当前深度图导出为图像文件"""
        if self.last_depths is None or self.last_angles is None:
            self.render_image()
        
        plt.figure(figsize=(10, 5))
        
        # 绘制条形图
        plt.subplot(1, 2, 1)
        angle_degrees = np.degrees(self.last_angles)
        plt.bar(angle_degrees, self.last_depths, width=np.degrees(self.robot['resolution']), align='center')
        plt.xlabel('角度 (度)')
        plt.ylabel('深度')
        plt.title('深度图')
        
        # 绘制极坐标图
        plt.subplot(1, 2, 2, projection='polar')
        plt.polar(self.last_angles, self.last_depths)
        plt.title('极坐标深度图')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"深度图已保存至 {filename}")
    
    def export_environment(self, filename="environment.png"):
        """将当前环境导出为图像文件"""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.environment.T, cmap='binary', origin='lower')
        
        # 绘制机器人位置和朝向
        plt.plot(self.robot['y'], self.robot['x'], 'ro', markersize=8)
        dir_x = self.robot['x'] + 5 * np.cos(self.robot['theta'])
        dir_y = self.robot['y'] + 5 * np.sin(self.robot['theta'])
        plt.plot([self.robot['y'], dir_y], [self.robot['x'], dir_x], 'r-')
        
        # 绘制轨迹
        traj_x = [pos[0] for pos in self.trajectory]
        traj_y = [pos[1] for pos in self.trajectory]
        plt.plot(traj_y, traj_x, 'b--', alpha=0.7)
        
        plt.title('环境地图')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"环境地图已保存至 {filename}")

# 示例：如何使用Simulator类
if __name__ == "__main__":
    # 创建模拟器实例
    sim = Simulator(size=(100, 100), window_size=(1200, 800))
    
    while sim.running:
        # 处理事件
        sim.handle_events()
        
        # 获取最新深度图 depth and cooresponding angles both len=90
        depth, angle = sim.render_image()
        
        # robot_pose
        x, y, theta = sim.get_robot_pose()
        
        print(depth, np.rad2deg(angle))
        print(x, y, theta)
        
        sim.plot_sim()
       
        
        # 更新显示
        pygame.display.flip()
        
        # 控制帧率
        sim.clock.tick(30)