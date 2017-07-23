# -*- coding: utf-8 -*-
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib
from Config import Config
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from myUtils import lazy_property, dense

GLOBAL_RUNNING_R = []  # 历史成绩记录

class ACNet(object):
    def __init__(self, scope, config, globalAC=None):
        self.config = config
        self.globalAC = globalAC
        self.action_dim = self.config.N_A
        self.state_dim = self.config.N_S

        if scope == self.config.GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], 'S')
                self.a_prob, self.v
                self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
                self.action_input = tf.placeholder(tf.int32, [None, ])
                self.v_input = tf.placeholder(tf.float32, [None, 1])

                self.a_prob, self.v
                self.choose_action

                self.TD_loss
                self.critic_loss
                self.actor_loss

                self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

                self.optimizer_actor = tf.train.RMSPropOptimizer(self.config.LR_A, name='RMSPropA')
                self.optimizer_critic = tf.train.RMSPropOptimizer(self.config.LR_C, name='RMSPropC')
                self.pull_params
                self.push_params

    @lazy_property
    def critic_loss(self):
        return tf.reduce_mean(tf.square(self.TD_loss))

    @lazy_property
    def actor_loss(self):
        log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.action_input, self.action_dim, dtype=tf.float32),
                                 axis=1, keep_dims=True)
        # TODO: encourage exploration
        exp_v = log_prob * self.TD_loss
        entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1, keep_dims=True)  # encourage exploration
        exp_v = self.config.ENTROPY_BETA * entropy + exp_v
        return tf.reduce_mean(-exp_v)

    @lazy_property
    def TD_loss(self):
        return tf.subtract(self.v_input, self.v)

    @lazy_property
    def v(self):
        with tf.variable_scope('critic'):
            w_i = tf.random_uniform_initializer(0., 0.1)
            b_i = tf.zeros_initializer()
            with tf.variable_scope('dense1'):
                dense1 = dense(self.state_input, 100, [100], w_i, activation=tf.nn.relu6)
            with tf.variable_scope('dense2'):
                dense2 = dense(dense1, 1, [1], w_i, b_i, activation=None)
            return dense2

    @lazy_property
    def a_prob(self):
        with tf.variable_scope('actor'):
            w_i = tf.random_uniform_initializer(0., 0.1)
            b_i = tf.zeros_initializer()
            with tf.variable_scope('dense1'):
                dense1 = dense(self.state_input, 200, None, w_i, b_i, activation=tf.nn.relu6)
            with tf.variable_scope('dense2'):
                dense2 = dense(dense1, self.action_dim, None, w_i, b_i, activation=tf.nn.softmax)
            return dense2

    @lazy_property
    def pull_params(self):
        pull_actor_params = [tf.assign(l_p, g_p) for g_p, l_p in zip(self.globalAC.actor_params, self.actor_params)]
        pull_critic_params = [tf.assign(l_p, g_p) for g_p, l_p in zip(self.globalAC.critic_params, self.critic_params)]
        return [pull_actor_params, pull_critic_params]

    @lazy_property
    def push_params(self):  # 注意：push操作不是assign了，而是在globalAC上应用lobal的梯度
        push_actor_params = self.optimizer_actor.apply_gradients(zip(self.actor_grads, self.globalAC.actor_params))
        push_critic_params = self.optimizer_critic.apply_gradients(zip(self.critic_grads, self.globalAC.critic_params))
        return [push_actor_params, push_critic_params]

    @lazy_property
    def choose_action(self):
        return tf.multinomial(tf.log(self.a_prob), 1)[0][0]  # 莫名其妙，不加tf.log可能出现超出action_dim的值


class Worker(object):
    def __init__(self, name, globalAC, config):
        self.config = config
        self.env = gym.make(self.config.GAME).unwrapped  # 取消-v0的限制，成绩可以很大
        self.name = name
        self.AC = ACNet(name, config, globalAC)


    def work(self):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and self.config.GLOBAL_EP < self.config.MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                a = SESS.run(self.AC.choose_action, feed_dict={self.AC.state_input: [s]})
                s_, r, done, info = self.env.step(a)
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(-5 if done else r)
                # update global and assign to local net
                if total_step % self.config.UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done else SESS.run(self.AC.v, {self.AC.state_input: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + self.config.GAMMA * v_s_
                        buffer_v_target.append([v_s_])
                    buffer_v_target.reverse()
                    SESS.run(self.AC.push_params, feed_dict={
                        self.AC.state_input: buffer_s,
                        self.AC.action_input: buffer_a,
                        self.AC.v_input: buffer_v_target,
                    })
                    SESS.run(self.AC.pull_params)
                    buffer_s, buffer_a, buffer_r = [], [], []
                s = s_
                total_step += 1
                if done:
                    GLOBAL_RUNNING_R.append(ep_r if not GLOBAL_RUNNING_R else 0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print '{:4} Ep: {:4}  Reward: {:4}   GLOBAL_RUNNING_R: {}'\
                        .format(self.name, self.config.GLOBAL_EP, int(ep_r), int(GLOBAL_RUNNING_R[-1]))
                    self.config.GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()
    unique_config = Config()
    with tf.device("/cpu:0"):
        GLOBAL_AC = ACNet(Config.GLOBAL_NET_SCOPE, unique_config)  # we only need its params
    workers = []
    for i in range(Config.N_WORKERS):
        i_name = 'W_%i' % i  # worker name
        workers.append(Worker(i_name, GLOBAL_AC, unique_config))
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    # tensorboard
    # if Config.OUTPUT_GRAPH:
    #     if os.path.exists(Config.LOG_DIR):
    #         shutil.rmtree(Config.LOG_DIR)
    #     tf.summary.FileWriter(Config.LOG_DIR, SESS.graph)
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=lambda: worker.work())
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()