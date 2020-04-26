#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
import copy
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, from_euler_angles
import quaternion as qn # uses (w,x,y,z)
from rlbench.environment import Environment # uses(x,y,z,w)
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from pyrep.const import ConfigurationPathAlgorithms as Algos

from rlbench_example import *


class ResetAgent(object):
    def __init__(self, env=None, task=None):
        super().__init__()
        if env is None:
            raise Exception("Environment not provided.")
        if task is None:
            raise Exception("Task not provided.")
            
        self.env = env
        self.task = task
        self.sensor = NoisyObjectPoseSensor(self.env)
