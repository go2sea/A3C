# -*- coding: utf-8 -*_
import gym
import multiprocessing


class Config:
    GAME = 'CartPole-v0'
    OUTPUT_GRAPH = True
    LOG_DIR = './log'
    N_WORKERS = multiprocessing.cpu_count()
    MAX_GLOBAL_EP = 1000
    GLOBAL_EP = 0
    GLOBAL_NET_SCOPE = 'Global_Net'
    UPDATE_GLOBAL_ITER = 50
    GAMMA = 0.9
    ENTROPY_BETA = 0.001
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic

    env = gym.make(GAME)

    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n

