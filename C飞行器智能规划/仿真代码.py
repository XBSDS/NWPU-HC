                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   """
项目名称: 无人机集群协同突防与智能规划仿真
文件名: 最终.py
描述: 
    本项目基于Python与Matplotlib库，构建了一个多智能体无人机集群仿真环境。
    实现了无人机集群的集结、分路侦查、路径规划及协同突击功能。
    包含基于人工势场法(APF)的避障算法、基于轨迹插值的路径生成算法，以及具有动态追击和多目标打击能力的敌方防御系统。
    可视化界面采用了科幻风格的HUD设计，实时显示战术状态、威胁等级及血量信息。

姓名: 杜超卓
日期: 2025-12-23
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import platform
from collections import deque
import os
from matplotlib import font_manager
from matplotlib.patches import Ellipse
# ==========================================
# 0. 系统配置与字体初始化模块
# ==========================================
def init_fonts():
    """
    初始化字体配置，解决Matplotlib中文乱码问题。
    优先加载指定的'腾讯体'文件，若不存在则回退到系统默认中文字体。
    """
    # 指定自定义字体文件的路径
    font_path = r"E:\飞行器智能规划大作业\腾讯体.ttf"
    
    # 为了兼容性，如果绝对路径没找到，尝试在当前目录下寻找
    if not os.path.exists(font_path):
        font_path = "腾讯体.ttf" 
        
    if os.path.exists(font_path):
        # 如果找到了字体文件，将其注册到字体管理器中
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        custom_font_name = prop.get_name()
        
        # 强制设置全局字体族
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [custom_font_name, 'Microsoft YaHei', 'SimHei'] 
        plt.rcParams['font.monospace'] = [custom_font_name, 'Microsoft YaHei', 'SimHei']
        
        print(f"--> [系统] 字体加载成功: {custom_font_name}")
    else:
        # 如果未找到文件，使用系统自带的常见中文字体作为备选
        print("--> [警告] 未找到自定义字体文件，回退到系统默认配置。")
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']

    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图的基础分辨率
    plt.rcParams['figure.dpi'] = 100
    # 隐藏Matplotlib默认的工具栏，保持界面整洁
    plt.rcParams['toolbar'] = 'None'

# 执行字体初始化
init_fonts()

# ==========================================
# 1. 全局仿真参数配置
# ==========================================
GRID_SIZE = 100                 # 地图尺寸 100x100米
TARGET_POS = np.array([95.0, 50.0]) # 最终任务目标点坐标
COMM_RANGE = 15                 # 无人机之间的通信距离
SCOUT_SENSING_RANGE = 25.0      # 侦查机最大探测半径
ATTACK_SENSING_RANGE = 12.0     # 攻击机最大探测半径
MAX_STEPS = 2000                # 最大仿真步数（防止死循环）

# 人工势场法(APF)力学参数
K_ATTRACT = 0.5         # 引力系数：控制飞向目标的拉力
K_REPEL_HAZARD = 100.0  # 障碍物斥力系数：控制避障力度
K_REPEL_DRONE = 30.0    # 机间斥力系数：控制防碰撞力度

# 协同控制参数
WAYPOINT_REACH_DIST = 4.0   # 到达路径点的判定阈值（米）
FORMATION_SPACING = 4.0     # 编队飞行时的期望间距
SAFE_CORRIDOR_WIDTH = 8.0   # 安全通道的逻辑宽度

# 界面UI调色板 (字典结构，方便统一管理颜色)
PALETTE = {
    'bg_main':    '#050a14',  # 主背景色（深蓝黑）
    'bg_panel':   '#0a121f',  # 仪表盘背景
    'card_bg':    '#0d1b2a',  # 信息卡片背景
    'grid_line':  '#1b2d42',  # 网格线颜色
    'text_main':  '#cceeff',  # 主文本色
    'text_sub':   '#5588aa',  # 副文本色
    'cyan_neon':  '#00f0ff',  # 霓虹青（高亮/装饰）
    'blue_core':  '#0088ff',  # 核心蓝（我方单位）
    'magenta':    '#d600ff',  # 紫红
    'red_warn':   '#ff0055',  # 警告红（敌方/高危）
    'orange_haz': '#ff6600',  # 警戒橙
    'green_safe': '#00ffaa',  # 安全绿
    'yellow':     '#ffdd00',  # 提示黄
    'border':     '#2a4d69'   # 边框色
}

# ==========================================
# 2. 类定义：敌方防御单位 (DynamicFireZone)
# ==========================================
class DynamicFireZone:
    """
    敌方火力点类。
    包含固定炮塔和移动巡逻单位，具有探测、攻击和动态追击能力。
    """
    def __init__(self, x, y, radius, lethality, is_mobile=True):
        self.pos = np.array([x, y], dtype=float) # 当前位置
        self.lethality = lethality               # 攻击命中率
        self.is_active = True                    # 存活状态
        self.is_mobile = is_mobile               # 是否为移动单位
        
        # 根据单位类型初始化属性
        if self.is_mobile:
            self.max_hp = 2             # 移动单位血量
            self.hp = 2
            self.radius = 5.0           # 攻击半径
            self.detection_range = 5.0  # 侦查半径（近视）
            self.influence_dist = 3.0   # 势场影响距离
            self.speed = 0.15           # 移动速度
            # 随机生成一个初始巡逻目标点
            self.target_waypoint = np.array([random.uniform(20, 80), random.uniform(20, 80)])
        else:
            self.max_hp = 5             # 固定单位血量（高护甲）
            self.hp = 5
            self.radius = 8.0           # 攻击半径（大范围）
            self.detection_range = 20.0 # 侦查半径（广域视野）
            self.influence_dist = 5.0   # 势场影响距离
            
        # 闪烁周期（用于视觉特效，不影响逻辑）
        self.flicker_period = random.randint(60, 120)

    def update(self, step, drones=None, global_alert_pos=None):
        """
        更新火力点的状态和位置。
        包含智能追击逻辑：如果接收到全局警报或发现目标，则进行移动。
        """
        # 如果血量归零，标记为非激活状态，停止更新
        if self.hp <= 0:
            self.is_active = False
            return

        self.is_active = True 
        
        # 仅移动单位需要更新位置
        if self.is_mobile:
            chase_target = None
            
            # 逻辑优先级1：优先追击自身视野内的目标
            if drones:
                min_dist = self.detection_range 
                closest_drone = None
                # 遍历所有存活无人机，寻找最近的
                for d in drones:
                    if d.status == "ALIVE":
                        dist = np.linalg.norm(d.pos - self.pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_drone = d
                # 如果发现目标，锁定位置
                if closest_drone:
                    chase_target = closest_drone.pos
            
            # 逻辑优先级2：响应全局警报（联动机制）
            # 如果自己没看见，但固定炮塔发现了目标并广播了位置，则前往支援
            if chase_target is None and global_alert_pos is not None:
                chase_target = global_alert_pos
            
            # 执行移动逻辑
            if chase_target is not None:
                # 追击模式：向目标全速移动
                direction = chase_target - self.pos
                dist = np.linalg.norm(direction)
                if dist > 0.1: # 防止重叠计算错误
                    self.pos += (direction / dist) * self.speed
            else:
                # 巡逻模式：向随机巡逻点移动
                direction = self.target_waypoint - self.pos
                dist = np.linalg.norm(direction)
                # 到达巡逻点后，生成新的随机点
                if dist < 1.0:
                    self.target_waypoint = np.array([random.uniform(20, 80), random.uniform(20, 80)])
                else:
                    self.pos += (direction / dist) * self.speed

# ==========================================
# 3. 类定义：我方无人机 (Drone)
# ==========================================
class Drone:
    """
    无人机单体类。
    区分侦查型(SCOUT)和攻击型(ATTACK)，具有不同的速度、血量和感知能力。
    """
    def __init__(self, drone_id, start_pos, squad_id=0, drone_type="ATTACK"):
        self.id = drone_id
        self.pos = np.array(start_pos, dtype=float) # 当前坐标
        self.drone_type = drone_type                # 类型：SCOUT 或 ATTACK
        self.status = "ALIVE"                       # 状态：ALIVE, DESTROYED, REACHED
        self.squad_id = squad_id                    # 小队编号
        self.path = [self.pos.copy()]               # 轨迹记录列表
        self.destroy_time = None                    # 坠毁时间（用于爆炸特效）
        self.explosion_pos = None                   # 坠毁位置
        
        # 初始化属性参数
        if drone_type == "SCOUT":
            self.max_hp = 3
            self.hp = 3
            self.sensing_range = 12.0 # 侦查范围大
            self.speed = 0.25         # 速度快（用于快速穿插）
        else:
            self.max_hp = 2
            self.hp = 2
            self.sensing_range = 5.0  # 感知范围小
            self.attack_range = 5.0   # 攻击射程
            self.speed = 0.12         # 速度慢（大部队）
        
        # 协同控制状态变量
        self.mode = "MOVING"          # 当前行为模式
        self.current_waypoint = None  # 当前导航点
        self.waypoint_queue = deque() # 待执行的路径点队列
        self.last_pos = self.pos.copy() # 上一帧位置
        self.stuck_counter = 0        # 卡死检测计数器

    def move(self, force):
        """
        根据受力更新无人机位置。
        :param force: 计算出的合力向量 (numpy array)
        """
        if self.status != "ALIVE":
            return
        
        # 根据当前战术模式动态调整速度限制
        speed = self.speed
        if self.mode == "HOVERING": speed *= 0.1    # 悬停
        elif self.mode == "WAITING": speed *= 0.3   # 待命
        elif self.mode == "BREACHING": speed *= 1.2 # 突击（加速）
        elif self.mode == "SCOUTING": speed *= 1.0  # 侦查
        
        # 物理动力学模拟：限制最大速度
        mag = np.linalg.norm(force)
        if mag > speed:
            force = (force / mag) * speed
        
        # 更新位置
        self.last_pos = self.pos.copy()
        self.pos += force
        self.path.append(self.pos.copy()) # 记录轨迹
        
        # 卡死检测：如果移动距离极小，计数器增加
        if np.linalg.norm(self.pos - self.last_pos) < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

    def set_waypoint(self, waypoint):
        """设置单一当前目标点"""
        self.current_waypoint = np.array(waypoint)
        
    def add_waypoints(self, waypoints):
        """将一组路径点加入队列"""
        for wp in waypoints:
            self.waypoint_queue.append(np.array(wp))
    
    def update_waypoint(self):
        """
        更新导航逻辑：
        检查是否到达当前点，如果是，则从队列中取出下一个点。
        """
        # 如果当前无目标但队列有任务，取出一个
        if self.current_waypoint is None and len(self.waypoint_queue) > 0:
            self.current_waypoint = self.waypoint_queue.popleft()
        
        # 判断是否到达当前目标点
        if self.current_waypoint is not None:
            dist = np.linalg.norm(self.pos - self.current_waypoint)
            if dist < WAYPOINT_REACH_DIST:
                # 到达后，切换下一个
                if len(self.waypoint_queue) > 0:
                    self.current_waypoint = self.waypoint_queue.popleft()
                else:
                    self.current_waypoint = None # 任务完成

# ==========================================
# 4. 类定义：协同决策系统 (CoordinationSystem)
# ==========================================
class CoordinationSystem:
    """
    集群的大脑。
    负责管理战术阶段（集结、侦查、规划、突击），并处理路径生成算法。
    """
    def __init__(self, drones, hazards):
        self.drones = drones
        self.hazards = hazards
        # 分组管理
        self.scouts = [d for d in drones if d.drone_type == "SCOUT"]
        self.attacks = [d for d in drones if d.drone_type == "ATTACK"]
        
        # 初始状态：集结
        self.phase = "RALLYING"
        self.rally_point = np.array([12.0, 50.0]) # 集结点设置在左侧
        
        self.safe_corridors = []   # 存储发现的安全通道
        self.breach_waypoints = [] # 存储最终规划的路径点
        self.death_zones = []      # 记录阵亡位置（视为禁区）
        self.path_found = False    # 路径就绪标志
        
    def update(self, step):
        """
        有限状态机(FSM)更新逻辑，随时间推进切换战术阶段。
        """
        alive_scouts = [d for d in self.scouts if d.status == "ALIVE"]
        alive_attacks = [d for d in self.attacks if d.status == "ALIVE"]
        
        # 实时记录阵亡点
        for d in self.drones:
            if d.status == "DESTROYED" and not hasattr(d, 'death_recorded'):
                self.death_zones.append(d.pos.copy())
                d.death_recorded = True
                print(f"[警告] 单元 {d.id} 阵亡，标记危险区")

        # ========== 阶段0: 全员集结 (RALLYING) ==========
        if self.phase == "RALLYING":
            ready_count = 0
            # 统计到达集结点的数量
            for drone in self.drones:
                if drone.status == "ALIVE":
                    dist = np.linalg.norm(drone.pos - self.rally_point)
                    if dist < 15.0: 
                        ready_count += 1
            
            total_alive = len(alive_scouts) + len(alive_attacks)
            # 如果80%到位或超时，进入下一阶段
            if (total_alive > 0 and ready_count / total_alive > 0.80) or step > 150:
                self.phase = "SCOUTING"
                print(f"[指令] 集结完毕 (T={step})，派出侦查分队")
                # 攻击机待命，侦查机出动
                for d in alive_attacks: d.mode = "HOVERING"
                for d in alive_scouts:  d.mode = "MOVING"

        # ========== 阶段1: 派出侦查 (SCOUTING) ==========
        elif self.phase == "SCOUTING":
            for d in alive_attacks: d.mode = "HOVERING"
            for scout in alive_scouts: scout.mode = "SCOUTING"
            
            # 检查是否有侦查机成功抵达终点
            scouts_reached = [s for s in self.scouts if s.status == "REACHED"]
            scouts_at_target = len(scouts_reached) > 0
            
            # 触发条件：有侦查机到达 OR 侦查机全灭 OR 超时
            if scouts_at_target or len(alive_scouts) == 0 or step > 600:
                print(f"[情报] 侦查阶段结束。到达侦查机数: {len(scouts_reached)}")
                
                # 调用核心算法：分析侦查路径
                all_scouts = [d for d in self.drones if d.drone_type == "SCOUT"]
                self.analyze_scout_paths(all_scouts)
                
                if len(self.safe_corridors) > 0:
                    self.path_found = True
                    print(f"[情报] 路径计算完成，生成 {len(self.safe_corridors)} 条突防走廊")
                else:
                    print("[警告] 有效数据不足，启用应急备用路径")
                    self.plan_default_path()
                
                self.phase = "PLANNING"

        # ========== 阶段2: 路径规划 (PLANNING) ==========
        elif self.phase == "PLANNING":
            # 根据分析结果生成路径
            if self.path_found:
                self.plan_breach_path()
            else:
                self.plan_default_path()
            
            # 将生成的路径分配给攻击机群
            self.assign_waypoints_to_drones(alive_attacks)
            self.phase = "BREACHING"
            print(f"[指令] 攻击编队全速突击，剩余侦查机自由突击")

        # ========== 阶段3: 协同突击 (BREACHING) ==========
        elif self.phase == "BREACHING":
            # 全员切换至突击模式
            for drone in alive_attacks:
                drone.mode = "BREACHING"
            for drone in alive_scouts:
                drone.mode = "BREACHING"

    def analyze_scout_paths(self, scouts):
        """
        核心算法：基于侦查机实际飞出的轨迹，提取安全通道。
        """
        self.safe_corridors = []
        
        # 即使侦查机阵亡，其尸体前的轨迹也是有价值的
        if len(scouts) == 0:
            scouts = [d for d in self.drones if d.drone_type == "SCOUT"]
        
        for scout in scouts:
            if len(scout.path) < 10:
                continue
            
            path = np.array(scout.path)
            
            # 过滤掉起步阶段，只保留进入战区后(X>15)的轨迹点
            valid_points = path[path[:, 0] > 15]
            
            if len(valid_points) > 5:
                # 评估该路径的安全性（离障碍物越远分越高）
                safety_score = self.evaluate_path_safety(valid_points)
                
                # 保存整条轨迹信息，以便后续进行曲线插值
                self.safe_corridors.append({
                    'path_points': valid_points, 
                    'score': safety_score,
                    'scout_id': scout.id
                })
                
                avg_y = np.mean(valid_points[:, 1])
                print(f"  - 侦查机{scout.id} 轨迹已记录 (点数:{len(valid_points)}), 安全分={safety_score:.1f}")
        
        # 按安全分数降序排列，优先选择最安全的
        self.safe_corridors.sort(key=lambda c: c['score'], reverse=True)
    
    def evaluate_path_safety(self, path_points):
        """计算路径的累积安全分数"""
        safety = 100.0
        for point in path_points:
            for h in self.hazards:
                dist = np.linalg.norm(point - h.pos)
                # 如果距离小于安全阈值，扣分
                danger = max(0, h.radius + 5 - dist)
                safety -= danger * 2
        return max(0, safety)
    
    def plan_breach_path(self):
        """
        核心算法：路径拟合。
        从最佳侦查轨迹中提取数据，利用插值生成一条平滑的曲线路径。
        """
        self.breach_waypoints = []
        
        if len(self.safe_corridors) == 0:
            self.plan_default_path()
            return
        
        # 取出最佳轨迹点
        best_corridor = self.safe_corridors[0]
        raw_points = best_corridor['path_points']
        
        # --- 曲线插值算法 ---
        # 1. 按X坐标排序（确保函数单调性）
        sorted_indices = np.argsort(raw_points[:, 0])
        sorted_points = raw_points[sorted_indices]
        
        # 2. 去重（去除X坐标相同的点，np.interp要求X单调递增）
        unique_x, unique_indices = np.unique(sorted_points[:, 0], return_index=True)
        unique_y = sorted_points[unique_indices, 1]
        
        # 3. 重新采样：在起点(15)到终点(95)之间生成25个均匀分布的点
        target_xs = np.linspace(15, 95, 25) 
        
        # 4. 线性插值计算对应的Y值
        if len(unique_x) > 1:
            target_ys = np.interp(target_xs, unique_x, unique_y)
        else:
            # 数据不足时降级为直线
            target_ys = np.full_like(target_xs, unique_y[0] if len(unique_y)>0 else 50)
            
        # 5. 构建最终路径点列表
        self.breach_waypoints = []
        for x, y in zip(target_xs, target_ys):
            self.breach_waypoints.append([x, y])
            
        # 6. 强制修正终点，确保准确入库
        self.breach_waypoints[-1] = TARGET_POS
    
    def plan_default_path(self):
        """备用方案：生成简单的折线路径"""
        self.breach_waypoints = [[15, 50], [40, 70], [70, 50], TARGET_POS]
    
    def assign_waypoints_to_drones(self, drones):
        """将生成的路径点分配给每架攻击机，并增加编队偏移量"""
        for i, drone in enumerate(drones):
            # 计算偏移量：形成3行纵队
            offset_y = (i % 3 - 1) * FORMATION_SPACING
            offset_x = (i // 3) * FORMATION_SPACING * 0.5
            waypoints = []
            for wp in self.breach_waypoints:
                adjusted_wp = [wp[0] + offset_x, wp[1] + offset_y]
                waypoints.append(adjusted_wp)
            drone.add_waypoints(waypoints)

# ==========================================
# 5. 主仿真引擎 (SwarmSimulation)
# ==========================================
class SwarmSimulation:
    """
    仿真主类。
    负责初始化环境、执行物理步进、处理碰撞检测和战斗结算。
    """
    def __init__(self, num_drones=15, num_hazards=6):
        self.hazards = []
        # 初始化地图布局 (精心设计的关卡)
        # 上方防线
        self.hazards.append(DynamicFireZone(35, 75, radius=6.0, lethality=0.1, is_mobile=False))
        self.hazards.append(DynamicFireZone(75, 80, radius=5.5, lethality=0.1, is_mobile=False))
        # 下方防线
        self.hazards.append(DynamicFireZone(45, 25, radius=6.0, lethality=0.1, is_mobile=False))
        # 中间要塞
        self.hazards.append(DynamicFireZone(60, 50, radius=7.0, lethality=0.1, is_mobile=False))
        # 移动巡逻队 (3个)
        for _ in range(3): 
            self.hazards.append(DynamicFireZone(
                random.uniform(25, 65), random.uniform(30, 70),
                radius=4.5, lethality=0.1, is_mobile=True
            ))
        
        self.drones = []
        self._setup_drones(num_drones)
        self.coordination = CoordinationSystem(self.drones, self.hazards)
    
    def _setup_drones(self, num_drones):
        """初始化无人机群"""
        num_scouts = 3
        num_attacks = num_drones - num_scouts
        # 生成侦查机 (ID 0-2)
        for i in range(num_scouts):
            d = Drone(i, [random.uniform(5, 15), random.uniform(30, 70)], squad_id=0, drone_type="SCOUT")
            self.drones.append(d)
        # 生成攻击机 (ID 3+)
        for i in range(num_attacks):
            d = Drone(num_scouts + i, [random.uniform(0, 10), random.uniform(20, 80)], squad_id=1, drone_type="ATTACK")
            self.drones.append(d)
    
    def get_shared_vision(self, drone):
        """模拟通信网络，获取邻居共享的障碍物信息"""
        shared_hazards = {}
        # 查找通信范围内的队友
        neighbors = [n for n in self.drones 
                    if n.status == "ALIVE" and np.linalg.norm(drone.pos - n.pos) < COMM_RANGE]
        check_nodes = neighbors + [drone]
        
        # 汇总信息
        for node in check_nodes:
            for h in self.hazards:
                if h.is_active and np.linalg.norm(node.pos - h.pos) < node.sensing_range:
                    shared_hazards[id(h)] = {'pos': h.pos, 'radius': h.radius, 'inf_dist': h.influence_dist}
        return list(shared_hazards.values())
    
    def compute_hybrid_force(self, drone, shared_hazards, alive_drones):
        """
        核心控制算法：混合导航力场计算。
        结合人工势场法(APF)与有限状态机(FSM)，计算无人机下一帧的受力向量。
        """
        total_force = np.zeros(2)
        phase = self.coordination.phase
        target = TARGET_POS 
        
        # 1. 目标引力计算 (基于当前战术阶段)
        if phase == "RALLYING":
            target = self.coordination.rally_point
            k_attract = K_ATTRACT * 1.5
            
        elif phase == "SCOUTING":
            if drone.drone_type == "SCOUT" or drone.mode == "SCOUTING":
                # 侦查机分段导航策略
                if drone.pos[0] < 85.0:
                    # 分散推进：上中下三路
                    offset_y = (drone.id % 3 - 1) * 30.0 
                    virtual_target = np.array([95.0, 50.0 + offset_y])
                    target = virtual_target
                    k_attract = K_ATTRACT * 2.0 
                else:
                    # 末端冲刺：强制归中
                    target = TARGET_POS 
                    k_attract = K_ATTRACT * 4.0 
            else:
                # 攻击机待命
                target = self.coordination.rally_point
                k_attract = K_ATTRACT * 0.2
                
        elif phase == "BREACHING":
            if drone.drone_type == "ATTACK":
                # 攻击机跟随规划的路径点
                drone.update_waypoint()
                target = drone.current_waypoint if drone.current_waypoint is not None else TARGET_POS
                k_attract = K_ATTRACT * 1.5
            else:
                # 侦查机自由冲锋
                target = TARGET_POS
                k_attract = K_ATTRACT * 3.0
            
        else: 
            target = self.coordination.rally_point
            k_attract = K_ATTRACT * 0.5 
            drone.mode = "HOVERING"

        # 计算引力向量
        v2t = target - drone.pos
        dist_t = np.linalg.norm(v2t)
        if dist_t > 0:
            total_force += k_attract * (v2t / dist_t)

        # 2. 障碍物斥力计算 (APF核心)
        f_rep_haz = np.zeros(2)
        for h in shared_hazards:
            vec = drone.pos - h['pos']
            dist = np.linalg.norm(vec)
            dist_e = max(dist - h['radius'], 0.5) # 有效距离
            
            # 动态调整安全边际
            is_rushing = (phase == "BREACHING" and drone.drone_type == "SCOUT") or \
                         (phase == "SCOUTING" and drone.pos[0] > 85.0)
            safe_margin = 0.6 if is_rushing else (1.5 if phase == "RALLYING" else 1.2)
            check_dist = h['inf_dist'] * safe_margin
            
            # 如果进入危险区，施加斥力
            if dist_e < check_dist:
                # 径向斥力公式
                # 计算斥力大小：基于人工势场法(APF)的改进公式
                # (1/dist_e - 1/check_dist): 距离越近斥力越大，距离超过check_dist斥力为0
                # (1/dist_e**2): 斥力梯度的导数项，确保靠近障碍物时斥力急剧增加
                rep_mag = K_REPEL_HAZARD * (1/dist_e - 1/check_dist) * (1/dist_e**2)
                f_rep_haz += rep_mag * (vec/dist)
                
                # 切向力 (改进APF)：帮助无人机绕行而非卡死
                if phase != "HOVERING": 
                    tangent = np.array([-vec[1], vec[0]])
                    if np.dot(tangent, v2t) < 0: tangent = -tangent # 选择顺着目标方向的一侧
                    f_rep_haz += rep_mag * 1.2 * (tangent / dist)
        total_force += f_rep_haz
        
        # 3. 无人机间斥力 (防碰撞)
        f_rep_drone = np.zeros(2)
        dist_to_final = np.linalg.norm(drone.pos - TARGET_POS)
        repel_factor = 0.05 if dist_to_final < 8.0 else 1.0 # 终点附近允许重叠
        neighbors = 0
        center_of_mass = np.zeros(2)

        for n in alive_drones:
            if n.id == drone.id: continue
            v = drone.pos - n.pos
            d = np.linalg.norm(v)
            if 0 < d < 3.5:
                # 侦查机斥力稍小，允许穿插
                k_rep = K_REPEL_DRONE * 0.5 if drone.drone_type == "SCOUT" else K_REPEL_DRONE
                f_rep_drone += k_rep * (1/d - 1/3.5) * (1/d**2) * (v/d) * repel_factor
            
            # 凝聚力计算 (仅攻击机)
            if phase in ["RALLYING", "SCOUTING", "WAITING"] and drone.drone_type == "ATTACK" and n.drone_type == "ATTACK":
                if d < 20.0:
                    center_of_mass += n.pos
                    neighbors += 1
        total_force += f_rep_drone

        # 施加凝聚力 (让编队更紧凑)
        if neighbors > 0 and phase in ["RALLYING", "SCOUTING", "WAITING"] and drone.drone_type == "ATTACK":
            center_of_mass /= neighbors
            v_cohere = center_of_mass - drone.pos
            total_force += 0.5 * (v_cohere / (np.linalg.norm(v_cohere) + 0.1))

        # 4. 防卡死扰动
        if drone.stuck_counter > 15:
            # 施加垂直于目标的随机强力
            perp_force = np.array([-v2t[1], v2t[0]])
            perp_force /= (np.linalg.norm(perp_force) + 0.1)
            total_force += perp_force * 3.0 * random.choice([-1, 1])
        
        return total_force

    def step(self, t):
        """
        物理仿真单步执行函数。
        包含：全局警报计算、状态更新、移动计算、战斗结算。
        """
        active_drones = [d for d in self.drones if d.status == "ALIVE"]
        
        # --- 步骤1: 战术联动系统 ---
        # 计算全局警报位置（如果固定炮塔发现了目标，通知移动单位）
        global_alert_pos = None
        min_alert_dist = float('inf')
        
        for h in self.hazards:
            if h.is_active and not h.is_mobile: # 仅固定塔
                for d in active_drones:
                    dist = np.linalg.norm(d.pos - h.pos)
                    if dist < h.detection_range: # 进入20米侦查圈
                        if dist < min_alert_dist:
                            min_alert_dist = dist
                            global_alert_pos = d.pos.copy()
        
        # --- 步骤2: 更新环境 ---
        for h in self.hazards:
            h.update(t, self.drones, global_alert_pos)
        
        self.coordination.update(t)
        
        # --- 步骤3: 无人机运动更新 ---
        for d in active_drones:
            shared_vision = self.get_shared_vision(d)
            force = self.compute_hybrid_force(d, shared_vision, active_drones)
            d.move(force)
            
            # 判断是否抵达
            if np.linalg.norm(d.pos - TARGET_POS) < 5.0:
                d.status = "REACHED"

        # --- 步骤4: 战斗结算 (双向伤害) ---
        
        # A. 敌方火力打击
        for h in self.hazards:
            if not h.is_active: continue
            targets_in_range = []
            for d in active_drones:
                # 只有进入攻击范围(radius)才会被打
                if d.status == "ALIVE" and np.linalg.norm(d.pos - h.pos) < h.radius:
                    targets_in_range.append(d)
            
            if not targets_in_range: continue
            
            # 固定塔可同时打3个，移动单位只能打1个
            max_targets = 3 if not h.is_mobile else 1
            targets_in_range.sort(key=lambda d: np.linalg.norm(d.pos - h.pos))
            
            for d in targets_in_range[:max_targets]:
                # 命中概率 5%
                if random.random() < 0.05:
                    d.hp -= 1
                    if d.hp <= 0:
                        d.status = "DESTROYED"
                        d.destroy_time = t 
                        d.explosion_pos = d.pos.copy()

        # B. 我方无人机反击
        current_active = [d for d in self.drones if d.status == "ALIVE"]
        for d in current_active:
            if d.drone_type == "ATTACK":
                for h in self.hazards:
                    if not h.is_active: continue
                    # 攻击机射程内判定
                    if np.linalg.norm(d.pos - h.pos) < d.attack_range:
                        if random.random() < 0.05:
                            h.hp -= 1
                            if h.hp <= 0: h.is_active = False 
        
        # 返回存活+抵达的有效单位数
        return len([d for d in self.drones if d.status != "DESTROYED"])

# ==========================================
# 6. 可视化渲染模块
# ==========================================
def draw_dashboard(ax, step, sim, total_drones, alive_count):
    """
    绘制右侧的科幻风格仪表盘 (V5.1版 - 极简无干扰)。
    """
    ax.clear()
    ax.set_facecolor('#0b1016') 
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 左侧分割线 (防止自动缩放导致线条居中)
    ax.plot([0, 0], [0, 1], color=PALETTE['cyan_neon'], linewidth=1, transform=ax.transAxes)
    
    # 1. 任务时间
    sim_time = step * 0.1
    ax.text(0.08, 0.92, f"{sim_time:05.1f}", color=PALETTE['cyan_neon'], 
            fontsize=45, weight='bold', fontfamily='sans-serif', transform=ax.transAxes)
    ax.text(0.08, 0.89, "MISSION TIME / SECONDS", color=PALETTE['text_sub'], 
            fontsize=8, transform=ax.transAxes)
    
    # 分隔线
    ax.plot([0.08, 0.95], [0.86, 0.86], color=PALETTE['grid_line'], lw=1, transform=ax.transAxes)

    # 2. 战术阶段显示
    phase = sim.coordination.phase
    phase_map = {
        "RALLYING": ("全员集结", PALETTE['text_main']),
        "SCOUTING": ("前出侦查", PALETTE['yellow']),
        "PLANNING": ("路径计算", PALETTE['cyan_neon']),
        "WAITING":  ("战术待命", PALETTE['blue_core']),
        "BREACHING":("协同突击", PALETTE['red_warn'])
    }
    cn_text, p_color = phase_map.get(phase, (phase, 'white'))
    
    ax.text(0.08, 0.78, "CURRENT PHASE // 当前阶段", color=PALETTE['text_sub'], fontsize=8, transform=ax.transAxes)
    ax.text(0.08, 0.72, cn_text, color=p_color, fontsize=28, weight='bold', transform=ax.transAxes)
    
    # 动态心跳条
    pulse_len = 0.4 + 0.1 * np.sin(step * 0.2)
    ax.plot([0.08, 0.08 + pulse_len], [0.70, 0.70], color=p_color, lw=3, transform=ax.transAxes)

    # 3. 编队完整度 (粒子血条)
    survival_rate = alive_count / total_drones
    ax.text(0.08, 0.60, f"SQUAD INTEGRITY // 完整度 {int(survival_rate*100)}%", 
            color=PALETTE['text_sub'], fontsize=9, transform=ax.transAxes)
    
    bar_x_start = 0.08
    bar_y = 0.56
    block_width = 0.05
    gap = 0.01
    
    for i in range(15):
        is_active = i < (survival_rate * 15)
        c = PALETTE['green_safe'] if survival_rate > 0.6 else PALETTE['red_warn']
        if not is_active: c = '#1a2a3a'
        rect = plt.Rectangle((bar_x_start + i*(block_width+gap), bar_y), 
                           block_width, 0.025, color=c, transform=ax.transAxes)
        ax.add_patch(rect)

    # 4. 威胁等级
    active_threats = sum(1 for h in sim.hazards if h.is_active)
    ax.text(0.08, 0.45, "THREAT LEVEL // 威胁指数", color=PALETTE['text_sub'], fontsize=9, transform=ax.transAxes)
    t_color = PALETTE['orange_haz'] if active_threats > 0 else PALETTE['green_safe']
    ax.text(0.08, 0.35, f"{active_threats:02d}", color=t_color, fontsize=42, weight='bold', transform=ax.transAxes)
    ax.text(0.35, 0.39, "ACTIVE\nHOSTILES", color=PALETTE['text_sub'], fontsize=7, transform=ax.transAxes)
    ax.text(0.35, 0.36, "已激活目标", color=t_color, fontsize=9, transform=ax.transAxes)

    # 5. 系统日志
    ax.text(0.08, 0.25, "SYSTEM LOG // 系统日志", color=PALETTE['text_sub'], fontsize=8, transform=ax.transAxes)
    ax.plot([0.08, 0.95], [0.24, 0.24], color=PALETTE['grid_line'], lw=1, transform=ax.transAxes)
    
    reached_num = len([d for d in sim.drones if d.status == "REACHED"])
    flying_num = len([d for d in sim.drones if d.status == "ALIVE"])
    
    logs = [
        f"> 目标距离: {np.linalg.norm(TARGET_POS - sim.drones[0].pos):.1f}m",
        f"> 安全路径: {len(sim.coordination.safe_corridors)} 条可用",
        f"> 战损统计: {total_drones - alive_count} 架损毁",
        f"> 任务单元: {flying_num} 飞行 | {reached_num} 抵达"
    ]
    
    for i, log in enumerate(logs):
        ax.text(0.08, 0.20 - i*0.045, log, color=PALETTE['text_main'], 
                fontsize=9, alpha=0.9, transform=ax.transAxes)

    ax.text(0.95, 0.02, "TACTICAL_TERM_V5.1", color=PALETTE['grid_line'], fontsize=6, ha='right', transform=ax.transAxes)

def draw_circle_as_ellipse(ax, center, radius, **kwargs):
    """
    在任意比例的坐标轴上绘制显示为圆形的椭圆
    """
    # 获取当前坐标轴的数据范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 获取坐标轴的像素尺寸
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_pixels = bbox.width * fig.dpi
    height_pixels = bbox.height * fig.dpi
    
    # 计算数据单位到像素的转换比例
    data_width = xlim[1] - xlim[0]
    data_height = ylim[1] - ylim[0]
    
    # 像素/数据单位 比例
    x_scale = width_pixels / data_width
    y_scale = height_pixels / data_height
    
    # 计算椭圆的高度（保持圆形显示）
    # 如果 x_scale > y_scale，需要压缩高度
    ellipse_height = radius * 2 * (x_scale / y_scale)
    
    return Ellipse(center, radius*2, ellipse_height, **kwargs)

def draw_map(ax, sim, t):
    """
    绘制主战术地图。
    包含：网格、单位、轨迹、特效、血量标签。
    """
    ax.cla()
    ax.set_facecolor(PALETTE['bg_main'])
    
    # 绘制网格与水印
    ax.grid(True, which='major', color=PALETTE['grid_line'], linestyle='-', linewidth=0.8, alpha=0.3)
    ax.text(50, 50, "SECTOR-07 [控制区]", color=PALETTE['grid_line'], alpha=0.1, fontsize=50, weight='bold', ha='center', va='center')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    # ax.set_aspect('equal')
    ax.set_xticklabels([]); ax.set_yticklabels([])

    # 扫描线特效
    scan_x = (t * 0.5) % 100 
    ax.axvline(x=scan_x, color=PALETTE['cyan_neon'], linewidth=1.5, alpha=0.1)
    
    # 绘制集结点
    if sim.coordination.phase in ["RALLYING", "SCOUTING"]:
        rp = sim.coordination.rally_point
        ax.add_patch(draw_circle_as_ellipse(ax, rp, 8, color=PALETTE['cyan_neon'], fill=False, linestyle='-', alpha=0.6))
        # ax.add_patch(plt.Circle(rp, 12, color=PALETTE['cyan_neon'], fill=False, linestyle='--', alpha=0.3))
        ax.text(rp[0], rp[1]-15, "集结点 / RALLY", color=PALETTE['cyan_neon'], fontsize=8, ha='center', alpha=0.5)

    # 绘制规划的路径（绿色虚线）
    if sim.coordination.path_found and len(sim.coordination.breach_waypoints) > 0:
        waypoints = np.array(sim.coordination.breach_waypoints)
        ax.plot(waypoints[:, 0], waypoints[:, 1], '--', color=PALETTE['green_safe'], linewidth=2, alpha=0.8)
        ax.scatter(waypoints[:, 0], waypoints[:, 1], color=PALETTE['green_safe'], s=20, alpha=0.8)

    # 绘制目标点
    pulse = 1 + 0.2 * np.sin(t * 0.1)
    ax.scatter(TARGET_POS[0], TARGET_POS[1], color=PALETTE['green_safe'], s=100 * pulse, marker='+', zorder=10)

    # 绘制敌方火力点
    for i, h in enumerate(sim.hazards):
        if h.is_active:
            # 1. 攻击范围圈 (实线) - 使用椭圆
            ax.add_patch(draw_circle_as_ellipse(ax, h.pos, h.radius, 
                                              color=PALETTE['orange_haz'], 
                                              fill=False, linestyle='-', linewidth=1.5, alpha=0.9))
            ax.add_patch(draw_circle_as_ellipse(ax, h.pos, h.radius, 
                                              color=PALETTE['red_warn'], alpha=0.1)) 
            
            # 2. 侦查范围圈 (虚线，仅固定塔显示) - 使用椭圆
            if not h.is_mobile:
                ax.add_patch(draw_circle_as_ellipse(ax, h.pos, h.detection_range, 
                                                  color=PALETTE['orange_haz'], 
                                                  fill=False, linestyle=':', linewidth=0.8, alpha=0.3))
            
            # 中心叉号
            ax.scatter(h.pos[0], h.pos[1], marker='x', color=PALETTE['orange_haz'], s=35, linewidth=2, alpha=0.9)
            
            # 血量标签
            label_text = f"T-{i+1:02d} [HP:{h.hp}]"
            ax.text(h.pos[0], h.pos[1]+h.radius+2, label_text, 
                   color=PALETTE['orange_haz'], fontsize=7, alpha=0.9, weight='bold', ha='center')
        else:
            # 摧毁状态
            ax.scatter(h.pos[0], h.pos[1], marker='x', color='#333', s=20, alpha=0.5)
            ax.text(h.pos[0], h.pos[1]-3, "DESTROYED", color='#555', fontsize=6, ha='center')

    # 绘制无人机
    for d in sim.drones:
        if d.status == "DESTROYED":
            # 爆炸特效渲染
            if hasattr(d, 'destroy_time') and d.destroy_time is not None:
                explosion_age = t - d.destroy_time
                if explosion_age < 20:
                    radius = explosion_age * 0.8
                    alpha = max(0, 1 - explosion_age / 20)
                    ax.add_patch(plt.Circle(d.explosion_pos, radius, color=PALETTE['red_warn'], fill=False, linewidth=3, alpha=alpha))
                    ax.add_patch(plt.Circle(d.explosion_pos, radius * 0.6, color=PALETTE['orange_haz'], fill=True, alpha=alpha * 0.3))
            ax.scatter(d.pos[0], d.pos[1], marker='x', color='#555', s=25, alpha=0.5)
            continue
        
        c = PALETTE['cyan_neon'] if d.drone_type == "SCOUT" else PALETTE['blue_core']
        if d.status == "REACHED": c = PALETTE['green_safe']
        
        # 绘制尾迹
        if len(d.path) > 3:
            recent_path = np.array(d.path[-12:])
            ax.plot(recent_path[:,0], recent_path[:,1], color=c, lw=1.2, alpha=0.5)
        
        # 绘制本体
        marker = 'd' if d.drone_type == "SCOUT" else 'o'
        size = 35 if d.drone_type == "SCOUT" else 25
        ax.scatter(d.pos[0], d.pos[1], s=size*4, color=c, alpha=0.3, edgecolors='none')
        ax.scatter(d.pos[0], d.pos[1], marker=marker, color='white', edgecolors=c, lw=1.2, s=size, zorder=6)
        
        # 显示血量
        hp_color = PALETTE['green_safe'] if d.hp >= d.max_hp else (PALETTE['yellow'] if d.hp > 1 else PALETTE['red_warn'])
        ax.text(d.pos[0], d.pos[1]+2.5, f"HP:{d.hp}", color=hp_color, fontsize=6, ha='center', weight='bold')

def draw_final_trajectories(fig, sim, total_drones):
    """
    仿真结束后，绘制静态的全局复盘图。
    """
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_facecolor(PALETTE['bg_main'])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    
    ax.text(50, 105, "任务轨迹复盘 / TRAJECTORY REVIEW", color=PALETTE['cyan_neon'], fontsize=16, weight='bold', ha='center')
    ax.grid(True, color=PALETTE['grid_line'], linestyle='--', linewidth=0.5, alpha=0.3)
    
    # 绘制危险区
    for h in sim.hazards:
        ax.add_patch(plt.Circle(h.pos, h.radius, color=PALETTE['red_warn'], alpha=0.1))
        ax.add_patch(plt.Circle(h.pos, h.radius, color=PALETTE['orange_haz'], fill=False, linestyle='--', linewidth=1.5, alpha=0.5))
        ax.scatter(h.pos[0], h.pos[1], marker='x', color=PALETTE['orange_haz'], s=40, linewidth=2)
    
    # 绘制起终点
    start_pos = sim.drones[0].path[0]
    ax.scatter(start_pos[0], start_pos[1], marker='s', s=200, color=PALETTE['cyan_neon'], edgecolors='white', linewidth=2, label='起点', zorder=10)
    ax.scatter(TARGET_POS[0], TARGET_POS[1], marker='*', s=400, color=PALETTE['green_safe'], edgecolors='white', linewidth=2, label='目标', zorder=10)
    
    # 绘制所有路径
    for d in sim.drones:
        if len(d.path) < 2: continue
        path = np.array(d.path)
        if d.status == "DESTROYED": color = '#ff4444'; alpha = 0.4; linestyle = ':'
        elif d.status == "REACHED": color = PALETTE['green_safe']; alpha = 0.8; linestyle = '-'
        else: color = PALETTE['blue_core']; alpha = 0.6; linestyle = '-'
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=alpha, linewidth=1.5, linestyle=linestyle)
        ax.text(path[-1, 0] + 1.5, path[-1, 1] + 1.5, f"D{d.id}", color=color, fontsize=8, weight='bold', alpha=alpha)
    
    # 标记阵亡点
    if len(sim.coordination.death_zones) > 0:
        death_zones = np.array(sim.coordination.death_zones)
        ax.scatter(death_zones[:, 0], death_zones[:, 1], marker='X', s=150, color=PALETTE['red_warn'], edgecolors='white', linewidth=2, label='阵亡位置', zorder=8)
    
    # 绘制规划好的安全路径
    if len(sim.coordination.breach_waypoints) > 0:
        waypoints = np.array(sim.coordination.breach_waypoints)
        ax.plot(waypoints[:, 0], waypoints[:, 1], '--', color=PALETTE['yellow'], linewidth=2.5, alpha=0.6, label='规划路径')
    
    # 绘制统计面板
    reached_count = len([d for d in sim.drones if d.status == 'REACHED'])
    destroyed_count = len([d for d in sim.drones if d.status == 'DESTROYED'])
    stats_text = f"任务统计\n━━━━━━━━━━\n总出动: {total_drones} 架\n成功到达: {reached_count} 架\n阵亡损失: {destroyed_count} 架\n存活率: {(reached_count/total_drones)*100:.1f}%"
    ax.text(2, 98, stats_text, color=PALETTE['text_main'], fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor=PALETTE['bg_panel'], alpha=0.9, edgecolor=PALETTE['cyan_neon'], linewidth=2))
    
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, facecolor=PALETTE['bg_panel'], edgecolor=PALETTE['cyan_neon'])
    ax.set_xlabel('X 坐标 (战区深度)', color=PALETTE['text_sub'])
    ax.set_ylabel('Y 坐标 (战区宽度)', color=PALETTE['text_sub'])
    plt.tight_layout()
    plt.show()

# ==========================================
# 7. 程序入口 (Main)
# ==========================================
def main():
    # 设置黑色背景风格
    plt.style.use('dark_background')
    
    # 创建主窗口
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=PALETTE['bg_main'])
    try:
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+50+50") # 设置窗口位置
    except:
        pass

    # 设置布局网格 (左侧地图占3份，右侧仪表盘占1份)
    gs = gridspec.GridSpec(1, 4, width_ratios=[3, 3, 3, 2.8])
    ax_map = fig.add_subplot(gs[0, 0:3])
    ax_hud = fig.add_subplot(gs[0, 3])
    plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04, wspace=0.1)
    
    # 初始化仿真对象
    total_drones = 15
    sim = SwarmSimulation(num_drones=total_drones, num_hazards=8)
    
    # 开启交互模式以支持动态刷新
    plt.ion()
    RENDER_SKIP = 3  # 渲染跳帧数（提升运行速度）
    success_msg_printed = False

    # 主循环
    for t in range(MAX_STEPS):
        # 物理计算更新
        alive_count = sim.step(t)
        
        # 统计状态
        flying_drones = [d for d in sim.drones if d.status == "ALIVE"]
        reached_drones = [d for d in sim.drones if d.status == "REACHED"]
        num_flying = len(flying_drones)
        num_reached = len(reached_drones)
        
        # 渲染绘图
        if t % RENDER_SKIP == 0:
            draw_map(ax_map, sim, t)
            draw_dashboard(ax_hud, t, sim, total_drones, alive_count)
            plt.pause(0.0001)
        
        # 提示逻辑：主力到达提示
        if not success_msg_printed and num_reached >= total_drones * 0.8:
            print(f"\n[提示] 主力部队已抵达 ({num_reached}/{total_drones})，等待剩余单位归队...")
            success_msg_printed = True
            
        # 结束条件：所有飞机都已着陆或损毁
        if num_flying == 0:
            print(f"\n[结束] 所有单位行动结束。最终抵达: {num_reached}/{total_drones}")
            break
    
    # 绘制最终复盘图
    print("正在生成轨迹复盘图...")
    plt.ioff()
    draw_final_trajectories(fig, sim, total_drones)

if __name__ == "__main__":
    main()