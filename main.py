#!/usr/bin/env python
# coding: utf-8

from rlbench.environment import Environment # uses(x,y,z,w)
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from src.retry_reset_agent import RetryResetAgent
from src.simple_fwd_agent import SimpleFwdAgent
from src.container_reset_agent import ContainerResetAgent
from src.rl_fwd_agent import RlFwdAgent
# Setup Environment:
obs_config = ObservationConfig()
obs_config.set_all(True)
# obs_config.left_shoulder_camera.rgb = True
# obs_config.right_shoulder_camera.rgb = True
action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)

static_positions = True # This will ensure non-random initialization

env = Environment(action_mode, '', obs_config, static_positions=static_positions)
task = env.get_task(EmptyContainer)
task.reset()

# Execute FWD policy
fwd_agent = RlFwdAgent(env, task)
fwd_agent.empty_the_container()

# Reset the environment

# Reset Agent 1
# reset_agent = RetryResetAgent(env, task)
# reset_agent.reset_env()

# Reset Agent 2
container_reset_agent = ContainerResetAgent(env, task)
container_reset_agent.reset_env()

