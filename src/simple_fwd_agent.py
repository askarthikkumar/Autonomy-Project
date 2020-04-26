#!/usr/bin/env python
# coding: utf-8

import numpy as np
import traceback

from src.rlbench_example import *
from src.fwd_agent import FWDAgent

class SimpleFwdAgent(FWDAgent):
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

                angles = self.get_fwd_policy_angles(retry_count)
                
                self.go_to_pose(pose, orientation_euler=angles)
                
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

