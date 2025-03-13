#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import os
import re
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
        log_file = os.path.join(self.episode_dir, self.params['design_name'] + '_log.csv')
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
            return state, reward
        except Exception as e:
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
        # 权重参数
        w_opt = 0.7  # 优化目标的权重
        w_con = 0.3  # 约束条件的权重

        # 优化目标的改进程度
        opt_improvement = (self.lut_6 - lut_6) / self.lut_6  # 相对改进比例

        # 约束条件的满足情况
        max_levels = self.params["fpga_mapping"]["levels"]
        if levels <= max_levels:
            constraint_met = True
            con_improvement = (max_levels - levels) / max_levels  # 相对改进比例
        else:
            constraint_met = False
            con_improvement = - (levels - max_levels) / max_levels  # 相对违反程度

        # 奖励计算
        if constraint_met:
            # 如果约束条件满足，奖励由优化目标和约束条件的改进共同决定
            reward = w_opt * opt_improvement + w_con * con_improvement
        else:
            # 如果约束条件未满足，引入较大的惩罚
            penalty = -10  # 惩罚值
            reward = penalty + w_opt * opt_improvement

        # 奖励归一化（可选）
        # reward = max(-1, min(1, reward))  # 将奖励限制在 [-1, 1] 范围内
        return reward
    
    def _get_state(self, design_file):
        return extract_features(design_file, self.params['yosys_binary'], self.params['abc_binary'])
    
        # 新增状态克隆方法
    def clone_state(self, state):
        """创建当前状态的深拷贝（用于MCTS模拟）"""
        return np.copy(state)