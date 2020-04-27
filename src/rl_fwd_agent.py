import numpy as np
import traceback

from src.rlbench_example import *
from src.fwd_agent import FWDAgent

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
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines import DQN,DDPG

# quaternion packages
from quaternion import from_rotation_matrix, quaternion, from_euler_angles
import quaternion as qn # uses (w,x,y,z)

script_dir = os.path.dirname(__file__)

class RlFwdAgent(FWDAgent):
    def __init__(self, env=None, task=None):
        super().__init__(env=env, task=task)
        
        self.sensor = NoisyObjectPoseSensor(self.env)

        self.movable_objects = [shape for shape in self.objs_dict if 'Shape' in shape]
        
        self.start_bins = ['large_container']
        self.target_bins = ['small_container0']
        
        self.max_retry = 10
        self.pad_z = 0.3
        self.home_pose = self.get_objects()['large_container'].get_pose()
        self.home_pose[2] += self.pad_z
        path = os.path.join(script_dir,"../Models/Grasp_Model")
        self.model = DQN.load(path)
        self.n_actions = 8

    def get_obj_pose(self, obj_name):
        return self.sensor.get_poses()[obj_name]
    
    def get_fwd_policy_angles(self, retry_count):
        if retry_count == 0:
            angle_x = 0
            angle_y = np.pi
            angle_z = 0 

        elif retry_count > 0:
            angle_x = 0
            angle_y = np.pi
            angle_z = np.pi/2
        
        elif retry_count > 5:
            angle_x = 0
            # Pick a random angle between 180 +- 40
            theta_range = 40 * (np.pi/180)
            angle_y = np.random.uniform(np.pi-theta_range, np.pi+theta_range)
            angle_z = np.pi/2
        else:
            angle_x = 0
            angle_y = np.pi
            angle_z = 0
        
        return np.array([angle_x, angle_y, angle_z])
    
    def get_grasp_pose(self, theta):
        q = quaternion(0,1,0,0)
        cos = np.cos(theta/2)
        sin = np.sin(theta/2)
        p = quaternion(cos,0,0,sin)
        rot_qt = p*q
        print("Theta",rot_qt,cos,sin,theta/2)
        rot_arr = qn.as_float_array(rot_qt)
        rot_qt = [rot_arr[1], rot_arr[2], rot_arr[3], rot_arr[0]]
        return rot_qt

    def empty_the_container(self):
        
        target_bin = self.target_bins[0]
        start_bin = self.start_bins[0]
        
        start_bin_pose = self.objs_dict[start_bin].get_pose()
        start_bin_pose[2] += self.pad_z

        target_bin_pose = self.objs_dict[target_bin].get_pose()
        target_bin_pose[2] += self.pad_z

        move_objs = self.movable_objects
    
        for i, shape in enumerate(move_objs):
            
            is_picked = False
            retry_count = 0

            while not is_picked and retry_count < self.max_retry:
               
                # go back to home position
                self.go_to_pose(start_bin_pose)
                # move above object
                pose = self.get_obj_pose(shape)
                self.go_to_pose_quat(pose,pad=0.2)

                # get scene prediction
                obs = self.env._scene.get_observation().wrist_rgb
                action,_ = self.model.predict(obs)

                # calculate quat and move towards the object
                theta = (2*np.pi)*(action/self.n_actions)
                quat = self.get_grasp_pose(theta)
                self.go_to_pose_quat(pose, pad=0, quat=quat)
                
                # grasp the object
                is_picked = self.grasp(self.objs_dict[shape])
                
                if not is_picked:
                    retry_count += 1
                    print("Retry count: ", retry_count)
            
            # move to home position
            self.go_to_pose(start_bin_pose)
            
            # move above small container
            pose = self.get_obj_pose(target_bin)
            self.go_to_pose(pose)
            
            # release the object
            self.release(self.objs_dict[shape])
        
        self.go_to_pose(self.home_pose)

if __name__ == "__main__":
    path = os.path.join(script_dir,"../Models/Grasp_Model")
    model = DQN.load(path)
    action, _ = model.predict(np.random.random((128,128,3)))
    print(action)
