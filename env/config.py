import dataclasses
# 旨在简化数据类的定义，减少样板代码，并提供更好的代码可读性。这对于处理大量数据对象的情况特别有用。
import numpy as np

@dataclasses.dataclass
class VehicularEnvConfig:
    def __init__(self):
        # 道路信息
        self.road_range: int = 1200  # 道路长度
        self.road_width: int = 50  # 道路宽度

        # 时间信息
        self.time_slot_start: int = 0
        self.time_slot_end: int = 84

        # 任务信息相关（要处理的任务）
        self.Function_min_task_datasize=2#
        self.Function_max_task_datasize = 5 #
        self.Function_task_computing_resource: float = 300  # 任务计算资源300cycles/bit
        self.Function_min_task_delay: int = 20  # 任务的最小延迟20s
        self.Function_max_task_delay: int = 25  # 任务的最大延迟25s

        # 任务队列相关（每个卸载对象自己产生的任务，即自身到达任务）
        self.min_rsu_task_number: int = 2    #RSU最小任务个数
        self.max_rsu_task_number: int = 3  #RSU最大任务个数
        self.min_vehicle_task_number: int = 4    #车辆最小任务个数,用于生成初始任务的个数
        self.max_vehicle_task_number: int = 5   #车辆最大任务个数,用于生成初始任务的个数
        self.min_task_datasize: float = 2  # 2 MB 每个任务的最小数据大小
        self.max_task_datasize: float = 4  # 4 MB   每个任务的最大数据大小

        # 车辆相关
        self.min_vehicle_speed: int = 30 #车辆行驶的最小速度
        self.max_vehicle_speed: int = 40 #车辆行驶的最大速度
        self.min_vehicle_compute_ability: float =20000  #最小计算能力20000Mcycles/s
        self.max_vehicle_compute_ability: float =25000   #最大计算能力40000Mcycles/s
        self.vehicle_number = 10    #车辆个数
        self.seed = 1    #随机种子
        self.min_vehicle_y_initial_location:float =0    #y坐标最小值
        self.max_vehicle_y_initial_location: float =50  #y坐标最大值
        self.vehicle_x_initial_location:list=[0,self.road_range]#x坐标初始值
        # RSU相关
        self.rsu_number = 3  #RSU的个数
        self.min_rsu_compute_ability: float = 25000 # 最小计算能力25000Mcycles/s
        self.max_rsu_compute_ability: float = 30000  # 最大计算能力30000Mcycles/s
        self._rsu_x_location: dict = {"rsu_1": 200, "rsu_2": 600, "rsu_3": 1000}
        self._rsu_y_location: dict = {"rsu_1": 50, "rsu_2": 50, "rsu_3": 50}

        # 通信相关
        self.rsu_range:int =400 #RSU通信距离400m
        self.vehicle_range: int = 200   #车辆通信距离200m
        self.r2v_B:float=20 #R2V带宽：20Mbps
        self.v2v_B:float=40#V2V带宽:40Mbps
        self.rsu_p:float=50 #RSU发射功率：50w
        self.vehicle_p:float=10 #车发射功率： 10w
        self.w:float=0.001 #噪声功率𝜔：0.001 W/Hz
        self.k:float=30 #固定损耗𝐾：20-40db取30
        self.theta:int=2    #路径损耗因子𝜎：2-6取2
        self.r2r_onehop_time:float=8#r2r一跳传输时间8s
        self.c2r_rate:float=0.5#C-R传输速率：0.5mb/s
        self.cloud_compute_ability:float=1800  #cloud计算能力15000Mcycles/s
        self.min_transfer_rate:float=0.01    #最小传输速率：0.01mb/s
        self.rsu_connect_time:float=10000 #RSU之间的联通时间
        self.cloud_connect_time:float=10000   #R2C的连通时间

        #惩罚
        self.punishment=-50
        self.fed_alpha: float = 1.0
        self.fed_penalty: float = abs(self.punishment)

        # 训练提速相关
        self.rate_integration_points: int = 500
        self.use_fast_local_observation: bool = True
        self.use_fast_path_search: bool = True

        #环境相关
        self.action_size=(self.rsu_number+self.vehicle_number+1)** 3#动作空间
        # 状态空间的最大值
        self.high = np.array([np.finfo(np.float32).max for _ in range(self.rsu_number+self.vehicle_number)])
        # 状态空间的最小值
        self.low = np.array([0 for _ in range(self.rsu_number+self.vehicle_number)])
