from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_T1_CFG as ROBOT_CFG
from .tracking_env_cfg import TrackingEnvCfg

# 如果 booster_assets 里没定义 T1_ACTION_SCALE，这里手动设一个
T1_ACTION_SCALE = 0.5 

@configclass
class RoughWoStateEstimationEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1. 加载 T1 机器人模型
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_ACTION_SCALE
        
        # 2. 修正后的地形配置
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            curriculum=True,
            sub_terrains={
                # 修正：noise_step 必须大于 0，即使 range 是 (0,0)
                "flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.3, 
                    noise_range=(0.0, 0.0), 
                    noise_step=0.01  # 这里改为 0.01
                ),
                
                # 修正：确认类名为 HfPyramidSlopedTerrainCfg，参数为 platform_width
                "pyramid_slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.4,
                    slope_range=(0.0, 0.4), 
                    platform_width=2.0,     
                    border_width=0.25,
                ),
                
                # 随机粗糙地面
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.3,
                    noise_range=(0.01, 0.04),
                    noise_step=0.01,
                ),
            },
        )

@configclass
class PlayFlatWoStateEstimationEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 演示模式直接使用平面
        self.scene.terrain.terrain_type = "plane"
        self.scene.num_envs = 32