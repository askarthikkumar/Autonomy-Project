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
        
        self.max_retry = 10
        self.pad_z = 0.35
        self.start_bins = ['small_container0']
        self.target_bins = ['large_container']
        
    def get_obj_pose(self, obj_name):
        return self.sensor.get_poses()[obj_name]
    
    def get_retry_policy_angles(self, retry_count):
        if retry_count == 0:
            angle_x = 0
            angle_y = np.pi
            angle_z = 0 

        elif retry_count == 1:
            angle_x = 0
            angle_y = np.pi
            angle_z = np.pi/2
        
        elif retry_count > 1:
            angle_x = 0
            # Pick a random angle between 180 +- 40
            theta_range = 40 * (np.pi/180)
            angle_y = np.random.uniform(np.pi-theta_range, np.pi+theta_range)
            angle_z = np.pi/2
        
        elif retry_count > 6:
            # Pick a random angle between 180 +- 40
            theta_range = 40 * (np.pi/180)
            angle_x = np.random.uniform(np.pi-theta_range, np.pi+theta_range)

            # Pick a random angle between 180 +- 40
            theta_range = 40 * (np.pi/180)
            angle_y = np.random.uniform(np.pi-theta_range, np.pi+theta_range)
            angle_z = np.pi/2

        else:
            angle_x = 0
            angle_y = np.pi
            angle_z = 0
        
        return np.array([angle_x, angle_y, angle_z])

    def move_objs_to_container(self, objects, container):
        
        move_objs = objects
        target_bin = container

        target_bin_pose = self.get_obj_pose(target_bin)
        target_bin_pose[2] += self.pad_z
    
        for i, shape in enumerate(move_objs):
            
            is_picked = False
            retry_count = 0

            while not is_picked and retry_count < self.max_retry:
               
                pose = self.get_obj_pose(shape)
                pose[2] += self.pad_z

                self.go_to_pose(pose)
                pose[2] -= self.pad_z

                angles = self.get_retry_policy_angles(retry_count)

                self.go_to_pose(pose, orientation_euler=angles)

                # grasp the object
                is_picked = self.grasp(self.objs_dict[shape])
                
                if not is_picked:
                    retry_count += 1
                    print("Retry count: ", retry_count)
                    self.open_gripper()
            
            pose[2] += self.pad_z
            self.go_to_pose(pose)

            pose = self.get_obj_pose(target_bin)
            
            pose[2] += self.pad_z
            self.go_to_pose(pose)
            
            pose[2] -= self.pad_z
            self.go_to_pose(pose)
            
            # release the object
            self.release(self.objs_dict[shape])

            pose[2] += self.pad_z
            self.go_to_pose(pose)
        
        self.go_to_pose(target_bin_pose)

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
        
        saved_pose[2] += self.pad_z
        self.go_to_pose(saved_pose)

        target_bin_pose = self.get_obj_pose(target_bin)
        target_bin_pose[2] += self.pad_z
        self.go_to_pose(target_bin_pose)
    
        target_bin_pose[0] -= 2 * handle_length
        self.go_to_pose(target_bin_pose)
    
        self.go_to_pose(target_bin_pose, orientation_euler=[np.pi, np.pi, 0])
        
        self.go_to_pose(target_bin_pose, orientation_euler=[0, np.pi, 0])

        target_bin_pose[0] += handle_length
        self.go_to_pose(target_bin_pose)
        
        for i in range(10):
            self.go_to_pose(saved_pose)
        
        saved_pose[2] -= self.pad_z
        self.go_to_pose(saved_pose)

        self.release(self.objs_dict[start_bin])
    
        self.go_to_pose(target_bin_pose)

        move_manually = []
        for shape in self.movable_objects:
            if not (self.is_contained(shape, target_bin)):
                move_manually.append(shape)
        
        self.move_objs_to_container(move_manually, target_bin)

        print("Reset Complete!")

