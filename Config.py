# -*- coding: utf-8 -*_
import gym
import multiprocessing


class Config:
    mode = 'continuous'
    # mode = 'discrete'
    GAME = 'CartPole-v0' if mode == 'discrete' else 'Pendulum-v0'
    print 'Game:', GAME
    OUTPUT_GRAPH = True
    LOG_DIR = './log'
    N_WORKERS = multiprocessing.cpu_count()
    MAX_GLOBAL_EP = 1000 if GAME == 'CartPole-v0' else 800
    GLOBAL_EP = 0
    GLOBAL_NET_SCOPE = 'Global_Net'
    UPDATE_GLOBAL_ITER = 50 if GAME == 'CartPole-v0' else 5
    GAMMA = 0.9
    ENTROPY_BETA = 0.001 if GAME == 'CartPole-v0' else 0.01
    LR_A = 0.001 if GAME == 'CartPole-v0' else 0.0001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic

    env = gym.make(GAME)

    N_S = env.observation_space.shape[0]
    if mode == 'discrete':  # 注意：离散action的游戏的action_space没有shape属性
        N_A = env.action_space.n
    elif mode == 'continuous':  # 注意：在Pendulum-v0游戏中，action的shape为(1,)，是一个长度为1的列表
        N_A = env.action_space.shape[0]
        ACTION_BOUND = [env.action_space.low, env.action_space.high]
        ACTION_GAP = env.action_space.high - env.action_space.low
        MAX_EP_STEP = 400


