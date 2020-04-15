# to remove tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
import os,logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# RL packages
import gym
import time
from gym import spaces
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.layers as tf_layers
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines import DQN,DDPG
# RLBench Packages
from empty_container import *

dir_path = os.path.dirname(os.path.realpath(__file__))
OBJECT = "Shape1"
N_ACTIONS = 8

# make custom environment class to only read values. It should not directly change values in the other thread.
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, machine):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.machine = machine
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low = -10, high=10, shape = (1,17))
        self.min_force = 1

    def step(self, action):
        theta = (2*np.pi)*(action/N_ACTIONS)
        quat = self.machine.get_grasp_pose(theta)
        objs = self.machine.get_objects(True)
        pose = objs[OBJECT].get_pose()
        print("ACTION TAKEN", action)
        try:
            path = self.machine.move_to(pose=pose, pad=0, quat=quat)
            self.machine.execute(path)
            cond = self.machine.grasp(objs[OBJECT])
            path = self.machine.move_to(pose=pose, quat=quat, pad=0.20)
            self.machine.execute(path)
            reward, success = self.get_reward(cond)
            obs = self.make_obs()
            done = True
            info = {"success":success}
            print("REWARD",reward)
            return obs,reward,done,info
        except:
            print("COULD NOT FIND PATH")
            reward = 0
            obs=self.make_obs()
            info = {}
            info["success"] = False
            print("REWARD",reward)
            return obs,reward,True,info

    def get_reward(self, cond):
        if cond is False:
            return -1, False
        else:
            forces = self.machine.env._robot.gripper.get_joint_forces()
            for force in forces:
                if force < self.min_force:
                    return -1, False
            return 1, True
        
    def make_obs(self):
        '''
        State vector of length 17
        contains two poses, and length,width and height of object bb
        '''
        objs = self.machine.get_objects(True)
        gripper_pose = list(self.machine.env._robot.gripper.get_pose())
        obj_pose = list(objs[OBJECT].get_pose())
        obj_bb = objs[OBJECT].get_bounding_box()
        x_diff = obj_bb[0]-obj_bb[3]
        y_diff = obj_bb[1]-obj_bb[4]
        z_diff = obj_bb[2]-obj_bb[5]
        state = np.array(gripper_pose+obj_pose+[x_diff,y_diff,z_diff]).reshape(1,-1)
        return state

    def reset(self):
        self.machine.task.reset()
        objs = self.machine.get_objects(True)
        pose = objs[OBJECT].get_pose()
        path = self.machine.move_to(pose,pad=0.2)
        self.machine.execute(path)
        obs = self.make_obs()
        return obs

    def render(self, mode='human'):
        pass

    def close (self):
        pass

if __name__ == "__main__":
    machine = StateMachine()
    machine.initialize()
    env = CustomEnv(machine)
    model = DQN(MlpPolicy, env, verbose=1, learning_starts=64, batch_size=64, 
                exploration_fraction=0.3, target_network_update_freq=32, tensorboard_log=dir_path+'/Logs/')
    model.learn(total_timesteps=1000)
    model.save(dir_path+"/Models/Grasp_Model")