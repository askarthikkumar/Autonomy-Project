#!/usr/bin/env python
# coding: utf-8

import numpy as np

from src.rlbench_example import *
from src.base_agent import BaseAgent

class ResetAgent(BaseAgent):
    def __init__(self, env=None, task=None):
        super().__init__(env=env, task=task)
        
    def reset_env(self):
        raise NotImplementedError