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

import traceback

class BaseAgent(object):
    def __init__(self, env=None, task=None):
        
        if env is None:
            raise Exception("Environment not provided.")
        if task is None:
            raise Exception("Task not provided.")

        self.env = env
        self.task = task
        
        self.objs_dict = self.get_objects()
        
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
                