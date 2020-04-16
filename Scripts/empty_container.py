#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy as sp
import copy
from scipy.spatial.transform import Rotation
from quaternion import from_rotation_matrix, quaternion, from_euler_angles
import quaternion as qn # uses (w,x,y,z)
from rlbench.environment import Environment # uses(x,y,z,w)
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from pyrep.const import ConfigurationPathAlgorithms as Algos


def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])


def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)
        
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz


class RandomAgent:

    def act(self, obs):
        delta_pos = [(np.random.rand() * 2 - 1) * 0.005, 0, 0]
        delta_quat = [0, 0, 0, 1] # xyzw
        gripper_pos = [np.random.rand() > 0.5]
        return delta_pos + delta_quat + gripper_pos


class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [1e-10] * 3 #[0.005, 0.005, 0.005] 
        self._rot_scale = [1e-10] * 3 #[0.01] * 3

    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}

        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            pose[:3] += pos
            pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses

def contains(r1, r2):
    #whether r2 is within r1
    return r1[0] < r2[0] < r2[1] < r1[1] and r1[2] < r2[2] < r2[3] < r1[3] and r1[4] < r2[4] < r2[5] < r1[5]

def get_edge_points(obj_bbox,obj_mat):
    bbox_faceedge=np.array([[obj_bbox[0],0,0],
    [obj_bbox[1],0,0],
    [0,obj_bbox[2],0],
    [0,obj_bbox[3],0],
    [0,0,obj_bbox[4]],
    [0,0,obj_bbox[5]]]).T #3x6
    bbox_faceedge=np.vstack((bbox_faceedge,np.ones((1,6)))) #4x6
    box=(obj_mat@bbox_faceedge).T #6X3 face edge coords in world frame
    rect=min(box[:,0]),max(box[:,0]),min(box[:,1]),max(box[:,1]),min(box[:,2]),max(box[:,2]) #max/min along x,y,z world axes
    return rect


class StateMachine(object):
    def __init__(self):
        self.env=None
        self.home=None
        self.task=None
        self.sensor=None
        self.objs_dict=None
        
    def initialize(self):
        DATASET = ''
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.left_shoulder_camera.rgb = True
        obs_config.right_shoulder_camera.rgb = True
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
        self.env = Environment(
            action_mode, DATASET, obs_config, False)
        self.sensor = NoisyObjectPoseSensor(self.env)
        self.env.launch()
        self.task = self.env.get_task(EmptyContainer)
        self.task.reset()
        self.home = self.get_objects(False)['large_container'].get_pose()
        self.home[2]+=0.3
        # demos = task.get_demos(3, live_demos=live_demos)
        
        self.objs = self.get_objects()
        self.movable_objects = ['Shape', 'Shape1', 'Shape3']
        self.target_bins = ['small_container0']
        self.start_bins = ['large_container']
        self.max_retry = 5
        self.env._robot.gripper.set_joint_forces([50,50])
        
        
    def get_objects(self, graspable=False):
        if graspable:
            objs = self.task._task.get_graspable_objects()
            objs_dict = {}
            for item in objs:
                name = item.get_name()
                objs_dict[name] = item
            return objs_dict
        else:
            objs = self.env._scene._active_task.get_base().\
                    get_objects_in_tree(exclude_base=True, first_generation_only=False)
            objs_dict = {}
            for obj in objs:
                name = obj.get_name()
                objs_dict[name] = obj
            return objs_dict

    # Move above object
    def move_to(self, pose, pad=0.05, ignore_collisions=True, quat=np.array([0,1,0,0])):
        target_pose = np.copy(pose)
        obs = self.env._scene.get_observation()
        init_pose=obs.gripper_pose
        obs = self.env._scene.get_observation()
        init_pose=obs.gripper_pose
        target_pose[2]+=pad
        path=self.env._robot.arm.get_path(np.array(target_pose[0:3]),quaternion=quat, trials=1000,
                                                ignore_collisions=True, algorithm=Algos.RRTConnect)
        # TODO catch errors and deal with situations when path not found
        return path
    
    def grasp(self,obj):
        # open the grippers
        cond=False
        while not cond:
            cond=self.env._robot.gripper.actuate(1,0.1)
            self.env._scene.step()
        cond=False
        # now close
        while not cond:
            cond=self.env._robot.gripper.actuate(0,0.1)
            self.env._scene.step()
        cond = self.env._robot.gripper.grasp(obj)
        return cond
    
    def release(self, obj):
        cond=False
        while not cond:
            cond=self.env._robot.gripper.actuate(1,0.1)
            self.env._scene.step()
        self.env._robot.gripper.release()
    
    def execute(self, path):
        done=False
        path.set_to_start()
        while not done:
            done = path.step()
            a = path.visualize()
            self.env._scene.step()
        return done
    
    def go_to(self, pose, pad = 0.05, quat=np.array([0,1,0,0]), gripper_close=False):
        pose_cp = copy.copy(pose)
        pose_cp[2]+=pad
        pose_cp[3:]=quat
        wp = pose_cp.tolist()+[1]
        if gripper_close:
            wp = pose_cp.tolist()+[0]
        try:
            self.task.step(wp)
        except:
            print("Retrying with normal path planner")
            path=self.move_to(pose,pad,True,quat)
            self.execute(path)
        return

    def reset(self):
        self.task.reset()
    
    def is_within(self,obj1,obj2):
        #whether obj2 is within obj1
        obj1_bbox=obj1.get_bounding_box();obj1_mat=np.array(obj1.get_matrix()).reshape(3,4);
        obj2_bbox=obj2.get_bounding_box();obj2_mat=np.array(obj2.get_matrix()).reshape(3,4);
        obj1_rect= get_edge_points(obj1_bbox,obj1_mat)
        obj2_rect= get_edge_points(obj2_bbox,obj2_mat)
        return contains(obj1_rect,obj2_rect)
    
    def picking_bin_empty(self):
        '''
         Returns whether the picking bin is empty
        '''
        self.objs_dict=machine.get_objects()
        bin_obj=self.objs_dict['large_container'] #?Which one
        for obj_name in self.objs_dict.keys():
            if (not ('container' in obj_name)):
                if self.is_within(bin_obj,self.objs_dict[obj_name]):
                    return False
        return True
    def placing_bin_full(self):
        '''
         Returns whether the placing bin is full
        '''
        self.objs_dict=machine.get_objects()
        bin_obj=self.objs_dict['small_container1'] #?Which one
        for obj_name in self.objs_dict.keys():
            if (not ('container' in obj_name)):
                if not (self.is_within(bin_obj,self.objs_dict[obj_name])):
                    return False
        return True
    
    def get_all_shapes(self):
        objects = []

        for name in self.objs:
            if "Shape" in name:
                objects.append(name)
        print(objects)
        return objects
    
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
        
    def move_objects_to_target(self, target_bins, start_bins):
        
        target_bin = target_bins[0]
        start_bin = start_bins[0]
        
        start_bin_pose = self.objs[start_bin].get_pose()
        start_bin_pose[2]+=0.3

        target_bin_pose = self.objs[target_bin].get_pose()
        target_bin_pose[2]+=0.3

        '''
        move_objs = []
        for obj in machine.movable_objects:
            if self.is_within(target_bin, obj):
                move_objs.append(obj)
        print(move_objs)
        '''
        move_objs = machine.movable_objects
    
        for i, shape in enumerate(move_objs):
            
            cond = False
            retry_count = 0
            while not cond and retry_count < self.max_retry:
                theta = 2*np.pi*retry_count/self.max_retry
                quat = self.get_grasp_pose(theta)
                # go back to home position
                machine.go_to(start_bin_pose,0,gripper_close=False)
                # move above object
                objs_poses = machine.sensor.get_poses()
                # pose=objs[shape].get_pose()
                pose=objs_poses[shape]
                machine.go_to(pose,0, quat=quat,gripper_close=False)
                # grasp the object
                cond = machine.grasp(self.objs[shape])
                if not cond:
                    retry_count += 1
                    print("retry count: ", retry_count)
            # move to home position
            machine.go_to(start_bin_pose,0,gripper_close=True)
            print("Gripper joint forces",self.env._robot.gripper.get_joint_forces())
            # move above small container
            objs_poses=machine.sensor.get_poses()
            pose = objs_poses[target_bin]
            pose[0] += (i*0.04 - 0.04)
            machine.go_to(pose,0.05,gripper_close=True)
            # release the object
            machine.release(self.objs[shape])
        machine.go_to(machine.home,0,gripper_close=False)


if __name__ == "__main__":    
    machine = StateMachine()
    machine.initialize()
    print("HERE")
    machine.move_objects_to_target(machine.target_bins, machine.start_bins)
    machine.move_objects_to_target(machine.start_bins, machine.target_bins)

