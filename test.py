import gym
from gym import wrappers
import numpy as np
from policy import Policy
from value import NNValueFunction
import scipy.signal
from utils import plotLearning

import tensorflow as tf

env_name = 'MountainCarContinuous-v0'   
#env_name = 'LunarLanderContinuous-v2'   


num_episodes=1000
gamma = 0.995                 # Discount factor
delta = 0.005                 # D_KL target value
batch_size = 5                # Number of episodes per training batch
hid1_size = 32                # Size of the first hidden layer for value and policy NNs
eps = 0.2                     # Epsilon: parameter for eps-greedy choice of action
init_logvar = -1              # Initial policy natural log of variance


def init_gym(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, eps, animate=True):
    """Run single episode with option to animate.

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        eps: epsilon-greedy parameter
        animate: boolean, True uses env.render() method to animate episode

    Returns: 3-tuple of NumPy arrays and 1 float
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        totalReward
    """
    obs = env.reset()
    observes, actions, rewards = [], [], []
    done = False
    step = 0.0
    ###
    totalReward = 0
    ###
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        
        observes.append(obs)
        
        action = policy.sample(obs)
        
        actions.append(action)
        obs, reward, done, _ = env.step(action.flatten())
        rewards.append(reward)
        totalReward += reward
    print ("Reward "+str(totalReward))
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float32), totalReward)

env, obs_dim, act_dim = init_gym(env_name)
env = gym.make(env_name)
env.seed(11)

policy = Policy(obs_dim, act_dim, delta, hid1_size, init_logvar)

filepath = 'models/mountain.tf'
#filepath = 'models/lunar.tf'
policy.load_test(filepath)

episode = 0
n_batch = 0
reward_history = []
while episode < num_episodes:
    trajectories = run_episode(env, policy, eps)

    
