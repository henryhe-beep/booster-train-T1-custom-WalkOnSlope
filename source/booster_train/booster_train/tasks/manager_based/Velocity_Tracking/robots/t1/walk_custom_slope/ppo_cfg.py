from isaaclab.utils import configclass
from booster_train.tasks.manager_based.Velocity_Tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 20000 # 人形机器人速度追踪较难，建议训练时间长一些
    experiment_name = "t1_walk_custom_slope"
    logger = "tensorboard"