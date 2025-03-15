#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import os
import re
import math
import datetime
import numpy as np
from subprocess import check_output
from .features import extract_features

def log(message):
    print('[DRiLLS {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)

class FPGASession:
    """
    A class to represent a logic synthesis optimization session using ABC
    """
    def __init__(self, params):
        self.params = params

        self.action_space_length = len(self.params['optimizations'])
        self.observation_space_size = 9     # number of features

        self.iteration = 0
        self.episode = 0
        self.sequence = ['strash']
        self.lut_6, self.levels = float('inf'), float('inf')

        self.best_known_lut_6 = (float('inf'), float('inf'), -1, -1)
        self.best_known_levels = (float('inf'), float('inf'), -1, -1)
        self.best_known_lut_6_meets_constraint = (float('inf'), float('inf'), -1, -1)

        # logging
        self.log = None
    
    def __del__(self):
        if self.log:
            self.log.close()
    
    def reset(self):
        """
        resets the environment and returns the state
        """
        self.iteration = 0
        self.episode += 1
        self.lut_6, self.levels = float('inf'), float('inf')
        self.sequence = ['strash']
        self.episode_dir = os.path.join(self.params['playground_dir'], str(self.episode))
        if not os.path.exists(self.episode_dir):
            os.makedirs(self.episode_dir)
        
        # logging
        log_file = os.path.join(self.episode_dir, self.params['design_name'] + '_log_' + str(self.episode) + '.csv')
        if self.log:
            self.log.close()
        self.log = open(log_file, 'w')
        self.log.write('iteration, optimization, LUT-6, Levels, best LUT-6 meets constraint, levels, episode, iteration, best LUT-6, levels, episode, iteration, best levels,LUT-6, episode, iteration\n')

        state, _ = self._run()

        # logging
        self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(int(self.lut_6)), str(int(self.levels))]) + '\n')
        self.log.flush()

        return state
    
    def step(self, optimization):
        """
        accepts optimization index and returns (new state, reward, done, info)
        """
        self.sequence.append(self.params['optimizations'][optimization])
        new_state, reward = self._run()

        # logging
        if self.lut_6 < self.best_known_lut_6[0]:
            self.best_known_lut_6 = (int(self.lut_6), int(self.levels), self.episode, self.iteration)
        if self.levels < self.best_known_levels[0]:
            self.best_known_levels = (int(self.levels), int(self.lut_6), self.episode, self.iteration)
        if self.levels <= self.params['fpga_mapping']['levels'] and self.lut_6 < self.best_known_lut_6_meets_constraint[0]:
            self.best_known_lut_6_meets_constraint = (int(self.lut_6), int(self.levels), self.episode, self.iteration)
        self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(int(self.lut_6)), str(int(self.levels))]) + ', ' +
            '; '.join(list(map(str, self.best_known_lut_6_meets_constraint))) + ', ' + 
            '; '.join(list(map(str, self.best_known_lut_6))) + ', ' + 
            '; '.join(list(map(str, self.best_known_levels))) + '\n')
        self.log.flush()

        return new_state, reward, self.iteration == self.params['iterations'], None

    def get_best_result_episode(self):
        return self.best_known_lut_6_meets_constraint[2]

    def _run(self):
        """
        run ABC on the given design file with the sequence of commands
        """
        self.iteration += 1
        output_design_file = os.path.join(self.episode_dir, str(self.iteration) + '.v')
        output_design_file_mapped = os.path.join(self.episode_dir, str(self.iteration) + '-mapped.v')
    
        abc_command = 'read ' + self.params['design_file'] + '; '
        abc_command += ';'.join(self.sequence) + '; '
        abc_command += 'write ' + output_design_file + '; '
        abc_command += 'if -K ' + str(self.params['fpga_mapping']['lut_inputs']) + '; '
        abc_command += 'write ' + output_design_file_mapped + '; '
        abc_command += 'print_stats;'
    
        try:
            proc = check_output([self.params['abc_binary'], '-c', abc_command])
            # get reward
            lut_6, levels = self._get_metrics(proc)
            reward = self._get_reward(lut_6, levels)
            self.lut_6, self.levels = lut_6, levels
            # get new state of the circuit
            state = self._get_state(output_design_file)
            if self.levels > self.params['fpga_mapping']['levels'] or self.lut_6 >= \
                    self.best_known_lut_6_meets_constraint[0]:
                os.remove(output_design_file)
                os.remove(output_design_file_mapped)
            return state, reward
        except Exception as e:
            os.remove(output_design_file)
            os.remove(output_design_file_mapped)
            print(e)
            return None, None

    def _get_metrics(self, stats):
        """
        parse LUT count and levels from the stats command of ABC
        """
        line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        
        ob = re.search(r'lev *= *[0-9]+', line)
        levels = int(ob.group().split('=')[1].strip())
        
        ob = re.search(r'nd *= *[0-9]+', line)
        lut_6 = int(ob.group().split('=')[1].strip())

        return lut_6, levels

    def _get_reward(self, lut_6, levels):
        # 动态权重调整参数
        base_opt_weight = 1
        base_con_weight = 1
        constraint = self.params["fpga_mapping"]["levels"]

        # 动态惩罚系数（基于当前最佳值的比例）
        best_lut_6 = self.best_known_lut_6_meets_constraint[0]
        best_levels = self.best_known_levels[0]
        penalty = -0.01

        # 奖励衰减因子（鼓励更早改进）
        decay_factor = 0.95

        # 层级约束满足比例
        con_ratio = levels / constraint if levels > 0 else 1.0
        dynamic_opt_weight = base_opt_weight / (1 + con_ratio)
        dynamic_con_weight = base_con_weight * con_ratio

        # 改进程度计算（添加平滑项避免除零）
        opt_improvement = (self.lut_6 - lut_6) / (self.lut_6 + 1e-6)

        # 约束违反程度计算
        level_delta = levels - constraint
        if level_delta <= 0:
            con_improvement = (constraint - levels) / (constraint + 1e-6)
            constraint_met = True
        else:
            # 指数级惩罚增长
            con_improvement = - math.exp(level_delta / constraint)
            constraint_met = False

        # 综合奖励计算
        if constraint_met and opt_improvement > 0:
            reward = (dynamic_opt_weight * opt_improvement +
                      dynamic_con_weight * con_improvement)
        else:
            reward = penalty

        return reward * (decay_factor ** self.iteration)
    
    def _get_state(self, design_file):
        return extract_features(design_file, self.params['yosys_binary'], self.params['abc_binary'])
    
        # 新增状态克隆方法
    def clone_state(self, state):
        """创建当前状态的深拷贝（用于MCTS模拟）"""
        return np.copy(state)