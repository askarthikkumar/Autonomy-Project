#!/usr/bin/env python
# coding: utf-8

import numpy as np

from src.rlbench_example import *
from src.base_agent import BaseAgent

class FWDAgent(BaseAgent):
    def __init__(self, env=None, task=None):
        super().__init__(env=env, task=task)
        
    def empty_the_container(self):
        raise NotImplementedError