#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import math
import tensorflow as tf
import numpy as np
import datetime
import time
from .scl_session import SCLSession as SCLGame
from .fpga_session import FPGASession as FPGAGame

def log(message):
    print('[DRiLLS {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)


class Node:
    def __init__(self, state, parent=None):
        self.state = state        # 当前状态
        self.parent = parent      # 父节点
        self.children = []        # 子节点
        self.visits = 0           # 访问次数
        self.value = 0.0          # 节点价值
        self.action_probs = None  # 动作概率分布

class MCTS:
    def __init__(self, a2c_agent, exploration_weight=1.0):
        self.agent = a2c_agent
        self.exploration_weight = exploration_weight
    
    def search(self, root_state, num_simulations=100):
        root = Node(root_state)
        
        for _ in range(num_simulations):
            node = root
            # 选择阶段
            while node.children:
                node = self._select_child(node)
            
            # 扩展阶段
            if node.visits > 0:
                node = self._expand(node)
            
            # 模拟阶段
            value = self._simulate(node.state)
            
            # 回溯更新
            self._backpropagate(node, value)
        
        return self._choose_action(root)
    
    def _select_child(self, node):
        # 使用UCT算法选择子节点
        total_visits = sum(child.visits for child in node.children)
        log_visits = math.log(total_visits + 1e-8)
        
        def uct_score(child):
            exploit = child.value / (child.visits + 1e-8)
            explore = self.exploration_weight * math.sqrt(log_visits / (child.visits + 1e-8))
            return exploit + explore
        
        return max(node.children, key=uct_score)
    
    def _expand(self, node):
        # 使用A2C的策略网络生成动作概率
        action_probs = self.agent.session.run(
            self.agent.actor_probs,
            feed_dict={self.agent.state_input: [node.state]}
        )[0]
        
        # 创建子节点
        for action in range(self.agent.num_actions):
            # 这里需要实现状态转换的克隆逻辑（需在FPGASession中添加clone方法）
            new_state = self.agent.game.clone_state(node.state) 
            child = Node(new_state, parent=node)
            child.action_probs = action_probs
            node.children.append(child)
        
        return node.children[0]  # 返回第一个子节点进行模拟
    
    def _simulate(self, state):
        # 使用当前策略网络进行快速模拟
        action_probs = self.agent.session.run(
            self.agent.actor_probs,
            feed_dict={self.agent.state_input: [state]}
        )[0]
        action = np.random.choice(range(len(action_probs)), p=action_probs)
        value = self.agent.session.run(
            self.agent.state_value,
            feed_dict={self.agent.state_input: [state]}
        )[0][0]
        return value
    
    def _backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _choose_action(self, node):
        # 选择访问次数最多的动作
        visit_counts = [child.visits for child in node.children]
        return np.argmax(visit_counts)



class Normalizer():
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.n = tf.zeros(num_inputs)
        self.mean = tf.zeros(num_inputs)
        self.mean_diff = tf.zeros(num_inputs)
        self.var = tf.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = tf.identity(self.mean)
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = tf.clip_by_value(self.mean_diff/self.n, clip_value_min=1e-2, clip_value_max=1000000000)

    def normalize(self, inputs):
        obs_std = tf.sqrt(self.var)
        return (inputs - self.mean)/obs_std
    
    def reset(self):
        self.n = tf.zeros(self.num_inputs)
        self.mean = tf.zeros(self.num_inputs)
        self.mean_diff = tf.zeros(self.num_inputs)
        self.var = tf.zeros(self.num_inputs)

class A2C:
    def __init__(self, options, load_model=False, fpga_mapping=False):
        if fpga_mapping:
            self.game = FPGAGame(options)
        else:
            self.game = SCLGame(options)

        self.num_actions = self.game.action_space_length
        self.state_size = self.game.observation_space_size
        self.normalizer = Normalizer(self.state_size)

        self.state_input = tf.placeholder(tf.float32, [None, self.state_size])

        # Define any additional placeholders needed for training your agent here:
        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])
        self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ])

        self.state_value = self.critic()
        self.actor_probs = self.actor()
        self.loss_val = self.loss()
        self.train_op = self.optimizer()
        self.session = tf.Session()

         # 修改损失函数
        self.mcts_probs = tf.placeholder(tf.float32, [None, self.num_actions])
        # 添加MCTS初始化
        self.mcts = MCTS(self, exploration_weight=1.5)

        # model saving/restoring
        self.model_dir = options['model_dir']
        self.saver = tf.train.Saver()

        if load_model:
            self.saver.restore(self.session, self.model_dir)
            log("Model restored.")
        else:
            self.session.run(tf.global_variables_initializer())
        
        self.gamma = 0.99
        self.learning_rate = 0.01

    def optimizer(self):
        """
        :return: Optimizer for your loss function
        """
        return tf.train.AdamOptimizer(0.01).minimize(self.loss_val)        

    def critic(self):
        """
        Calculates the estimated value for every state in self.state_input. The critic should not depend on
        any other tensors besides self.state_input.
        :return: A tensor of shape [num_states] representing the estimated value of each state in the trajectory.
        """
        c_fc1 = tf.contrib.layers.fully_connected(inputs=self.state_input,
                                                num_outputs=10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    
        c_fc2 = tf.contrib.layers.fully_connected(inputs=c_fc1,
                                                num_outputs=1,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
        
        return c_fc2

    def actor(self):
        """
        Calculates the action probabilities for every state in self.state_input. The actor should not depend on
        any other tensors besides self.state_input.
        :return: A tensor of shape [num_states, num_actions] representing the probability distribution
            over actions that is generated by your actor.
        """
        a_fc1 = tf.contrib.layers.fully_connected(inputs=self.state_input,
                                                num_outputs=20,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    
        a_fc2 = tf.contrib.layers.fully_connected(inputs=a_fc1,
                                                num_outputs=20,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
        
        a_fc3 = tf.contrib.layers.fully_connected(inputs=a_fc2,
                                                num_outputs=self.num_actions,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    
        return tf.nn.softmax(a_fc3)

    def loss(self):
        """
        :return: A scalar tensor representing the combined actor and critic loss.
        """
        # critic loss
        advantage = self.discounted_episode_rewards_ - self.state_value
        critic_loss = tf.reduce_sum(tf.square(advantage))

        # actor loss    
        kl_divergence = tf.reduce_sum(
            self.mcts_probs * tf.log(self.mcts_probs / (self.actor_probs + 1e-8)),
            axis=1
        )
        actor_loss = tf.reduce_sum(kl_divergence * advantage)    
        # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.log(self.actor_probs), 
        #                                                           labels=self.actions)
        # actor_loss = tf.reduce_sum(neg_log_prob * advantage)
        
        # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actor_probs,
        #                                                          labels=self.actions)
        # policy_gradient_loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_)
        # return policy_gradient_loss
        
        return critic_loss + actor_loss

    def save_model(self):
        save_path = self.saver.save(self.session, self.model_dir)
        log("Model saved in path: %s" % str(save_path))

    def train_episode(self):
        """
        train_episode will be called several times by the drills.py to train the agent. In this method,
        we run t    he agent for a single episode, then use that data to train the agent.
        """
        state = self.game.reset()
        self.normalizer.reset()
        self.normalizer.observe(state)
        state = self.normalizer.normalize(state).eval(session=self.session)
        done = False
        
        episode_states = []
        episode_actions = []
        episode_mcts_probs = []  # 新增MCTS策略记录
        episode_rewards = []
        
        while not done:
            log('  iteration: ' + str(self.game.iteration))
            # 使用MCTS选择动作
            mcts_probs = self._get_mcts_probs(state)
            action = np.random.choice(range(self.num_actions), p=mcts_probs)
            
            # 记录MCTS策略
            episode_mcts_probs.append(mcts_probs)
            action_probability_distribution = self.session.run(self.actor_probs, \
                feed_dict={self.state_input: state.reshape([1, self.state_size])})
            action = np.random.choice(range(action_probability_distribution.shape[1]), \
                p=action_probability_distribution.ravel())
            new_state, reward, done, _ = self.game.step(action)
            
            # append this step
            episode_states.append(state)
            action_ = np.zeros(self.num_actions)
            action_[action] = 1
            episode_actions.append(action_)
            episode_rewards.append(reward)
            
            state = new_state
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state).eval(session=self.session)
        
        # Now that we have run the episode, we use this data to train the agent
        start = time.time()
        discounted_episode_rewards = self._compute_n_step_rewards(episode_rewards)
        
        _ = self.session.run(self.train_op, feed_dict={
            self.state_input: np.array(episode_states),
            self.actions: np.array(episode_actions),
            self.mcts_probs: np.array(episode_mcts_probs),  # 新增
            self.discounted_episode_rewards_: discounted_episode_rewards
        })
        end = time.time()
        log('Episode Agent Training Time ~ ' + str((start - end) / 60) + ' minutes.')
        
        self.save_model()
        
        return np.sum(episode_rewards)
    
    def _get_mcts_probs(self, state):
    # 获取MCTS策略分布
        root = self.mcts.search(state, num_simulations=50)
        visit_counts = np.array([child.visits for child in root.children])
        return visit_counts / np.sum(visit_counts)

    def _compute_n_step_rewards(self, rewards, gamma=0.99, n_step=5):
        """改进的N-step奖励计算"""
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0

        # 反向计算
        for i in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[i]
            # 每n步截断
            if i % n_step == 0:
                running_add = rewards[i]
            discounted[i] = running_add

        discounted -= np.mean(discounted)
        discounted /= (np.std(discounted) + 1e-8)
        return discounted

    def discount_and_normalize_rewards(self, episode_rewards):
        """
        used internally to calculate the discounted episode rewards
        """
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative
    
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
    
        discounted_episode_rewards = (discounted_episode_rewards - mean) / std
    
        return discounted_episode_rewards

