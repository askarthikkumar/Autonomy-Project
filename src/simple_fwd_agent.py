#!/usr/bin/env python
# coding: utf-8

import numpy as np

from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, from_euler_angles
import quaternion as qn # uses (w,x,y,z)

from rlbench.environment import Environment # uses(x,y,z,w)
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from pyrep.const import ConfigurationPathAlgorithms as Algos

from rlbench_example import *

import traceback

class SimpleFwdAgent(object):
    def __init__(self, env=None, task=None):
        super().__init__()
        if env is None:
            raise Exception("Environment not provided.")
        if task is None:
            raise Exception("Task not provided.")

        self.env = env
        self.task = task
        
        self.sensor = NoisyObjectPoseSensor(self.env)

        self.objs_dict = self.get_objects()

        self.movable_objects = [shape for shape in self.objs_dict if 'Shape' in shape]
        
        self.start_bins = ['large_container']
        self.target_bins = ['small_container0']
        
        self.max_retry = 10
        self.pad_z = 0.3
        self.home_pose = self.get_objects()['large_container'].get_pose()
        self.home_pose[2] += self.pad_z
        
    def get_objects(self):
        objs = self.env._scene._active_task.get_base().\
                get_objects_in_tree(exclude_base=True, first_generation_only=True)
        objs_dict = dict()

        for obj in objs:
            name = obj.get_name()
            objs_dict[name] = obj

        return objs_dict

    def get_path(self, pose, orientation_euler=[0, np.pi, 0], ignore_collisions=True):
        # TODO catch errors and deal with situations when path not found
        path = self.env._robot.arm.get_path(pose[0:3], euler=orientation_euler, 
                trials=1000, ignore_collisions=True, algorithm=Algos.RRTConnect)

        return path
    
    def grasp(self, obj):
        # open the grippers
        is_open = False
        while not is_open:
            is_open = self.env._robot.gripper.actuate(1,0.1)
            self.env._scene.step()
        
        is_closed = False
        # now close
        while not is_closed:
            is_closed = self.env._robot.gripper.actuate(0,0.1)
            self.env._scene.step()
        
        grasped = self.env._robot.gripper.grasp(obj)
        return grasped
    
    def release(self, obj):
        is_open = False
        while not is_open:
            is_open = self.env._robot.gripper.actuate(1,0.1)
            self.env._scene.step()

        released = self.env._robot.gripper.release()
        return released
    
    def execute_path(self, path):
        done = False
        try:          
            path.set_to_start()
            while not done:
                done = path.step()
                path.visualize()
                self.env._scene.step()
        except Exception as e:
            traceback.print_exc()
            done = False
        return done
    
    def go_to_pose(self, pose, orientation_euler=[0, np.pi, 0]):
        path = self.get_path(pose, orientation_euler=orientation_euler)
        reached = self.execute_path(path)
        return reached

    def get_obj_pose(self, obj_name):
        return self.sensor.get_poses()[obj_name]
                
    def sort_objects(self):
        
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
                

                if retry_count > 0:
                    # Pick a random angle between 180 +- 40
                    theta_range = 40 * (np.pi/180)
                    theta = np.random.uniform(np.pi-theta_range, np.pi+theta_range)
                else:
                    theta = np.pi

                self.go_to_pose(pose, orientation_euler=[0, theta, 0])
                
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

