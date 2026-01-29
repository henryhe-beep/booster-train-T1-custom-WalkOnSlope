from __future__ import annotations

from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 注意：这里我们同时引入 Isaac Lab 原生的 mdp 和 booster 特有的 mdp
import isaaclab.envs.mdp as mdp 
import booster_train.tasks.manager_based.beyond_mimic.mdp as booster_mdp

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """场景配置"""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    # 机器人配置（由子类 env_cfg.py 填充）
    robot: ArticulationCfg = MISSING
    
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

@configclass
class CommandsCfg:
    """指令配置 - 修复了 rel_scaling_range 错误"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.1, # 10% 的机器人保持静止，增加多样性
        # 修正点：使用 ranges 子类来定义速度范围
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),    # 前进速度范围 0~1 m/s
            lin_vel_y=(-0.2, 0.2),  # 左右平移范围
            ang_vel_z=(-0.5, 0.5),  # 旋转范围
        ),
        debug_vis=True,
    )

@configclass
class ActionsCfg:
    """动作配置"""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 观测项
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """奖励函数 - 已根据 Isaac Lab 最新 API 修正名称"""
    
    # 1. 追踪速度奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    )

    # 2. 姿态奖励 (针对你的报错进行的修正)
    # base_orientation_l2 改为 flat_orientation_l2
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    
    # 3. 高度奖励
    # 注意：某些版本中也可能叫 base_height_l2，如果报错请改为 root_height_l2
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 1.05})

    # 4. 动作平滑奖励
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 5. 惩罚非足端接触
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[r"^(?!.*foot_link).*$"]), 
            "threshold": 1.0
        },
    )

    # 6. 额外建议：掉头/平稳奖励（可选，增加双足稳定性）
    # 如果之后运行想让它更稳，可以开启下面的项：
    # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001)

@configclass
class TerminationsCfg:
    """终止条件"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 摔倒判定
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})
    root_height_below_minimum = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})

@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15