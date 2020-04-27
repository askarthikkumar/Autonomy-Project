#!/usr/bin/env python
# coding: utf-8

import numpy as np

from src.rlbench_example import *
from src.reset_agent import ResetAgent

class ContainerResetAgent(ResetAgent):
    def __init__(self, env=None, task=None):
        super().__init__(env=env, task=task)
        
        self.sensor = NoisyObjectPoseSensor(self.env)

        self.movable_objects = [shape for shape in self.objs_dict if 'Shape' in shape]
        
        self.pad_z = 0.35
        self.start_bins = ['small_container0']
        self.target_bins = ['large_container']
        
    def get_obj_pose(self, obj_name):
        return self.sensor.get_poses()[obj_name]
    
    def reset_env(self):
        handle_length = 0.15

        start_bin = self.start_bins[0]
        target_bin = self.target_bins[0]
        
        # move above object
        saved_pose = self.get_obj_pose(start_bin)
        saved_pose[0] -= handle_length
        self.go_to_pose(saved_pose)
        
        # TODO: Add a retry 
        is_picked = self.grasp(self.objs_dict[start_bin])
        
        target_bin_pose = self.get_obj_pose(target_bin)
        target_bin_pose[0] -= handle_length
        target_bin_pose[2] += self.pad_z
        self.go_to_pose(target_bin_pose)
    
        self.go_to_pose(target_bin_pose, orientation_euler=[np.pi+0.1, np.pi, 0])
        
        self.go_to_pose(target_bin_pose, orientation_euler=[0, np.pi, 0])

        saved_pose[2] += self.pad_z
        self.go_to_pose(saved_pose)

        saved_pose[2] -= self.pad_z
        self.go_to_pose(saved_pose)

        self.release(self.objs_dict[start_bin])
    
        self.go_to_pose(target_bin_pose)

