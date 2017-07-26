# A3C
### Tensorflow implementation of A3C, both discrete & continuous action space.

### The a3c.py provides 2 mode of a3c: discrete & continuous.
    Discrete: The space of actions is limited. I use CartPole-v0 for test.
    Continuous: The space of actions is unlimited, and the shape of action is usually al list. I use Pendulum-v0 for test.
#### You can change the mode in Config.py
    mode = 'continuous'
    # mode = 'discrete'
    GAME = 'CartPole-v0' if mode == 'discrete' else 'Pendulum-v0'
#### Ways to get the action-dimension are different between discrete-mode & continuous-mode:
    if mode == 'discrete':  # 注意：离散action的游戏的action_space没有shape属性
        N_A = env.action_space.n
    elif mode == 'continuous':  # 注意：在Pendulum-v0游戏中，action的shape为(1,)，是一个长度为1的列表
        N_A = env.action_space.shape[0]
#### The result on Pendulum-v0:
![figure_1](/images/Pendulum_result.png)




