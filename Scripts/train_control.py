import gym
import rlbench.gym
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

env = gym.make("empty_container-state-v0",render_mode="human")
model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=dir_path+'/Logs/')
model.learn(total_timesteps=1000)
model.save("sac_ec")