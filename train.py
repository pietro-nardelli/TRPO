#! /usr/bin/env python3
"""
TRPO: Trust Region Policy Optimization
"""


import gym
from gym import wrappers
import numpy as np
from policy import Policy
from value import NNValueFunction
import scipy.signal
from utils import plotLearning

env_name = 'MountainCarContinuous-v0'   
num_episodes=5000
gamma = 0.995                   # Discount factor
lam = 0.98                      # Lambda for Generalized Advantage Estimation
kl_targ = 0.001                 # D_KL target value
batch_size = 5                 # Number of episodes per training batch
hid1_size = 32                  # Size of the first hidden layer for value and policy NNs
init_logvar = -1.0              # Initial policy natural log of variance


def init_gym(env_name):
    env = gym.make('MountainCarContinuous-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, animate=True):
    """Run single episode with option to animate.

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
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
        step += 1e-3  # increment time step feature
        totalReward += reward
        
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float32), totalReward)


def run_policy(env, policy, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, totalReward = run_episode(env, policy)
        # print(observes.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(unscaled_obs.shape)
        # print(observes.dtype)
        # print(actions.dtype)
        # print(rewards.dtype)
        # print(unscaled_obs.dtype)

        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards}
        trajectories.append(trajectory)
    
    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values.flatten()


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew

def avg_batch_rewards (trajectories):
    """ Calculate avarage of rewards in the trajectories batch """
    total_reward = []
    for i in range(len(trajectories)):        
        total_reward.append(np.sum(trajectories[i]['rewards']))
    avg = np.mean(total_reward)
    return avg

""" Main training loop

Args:
    env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
    num_episodes: maximum number of episodes to run
    gamma: reward discount factor (float)
    lam: lambda from Generalized Advantage Estimate
    kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
    batch_size: number of episodes per policy training batch
    hid1_size: hid1 size for policy and value_f
    init_logvar: natural log of initial policy variance
"""
env, obs_dim, act_dim = init_gym(env_name)
env = gym.make(env_name)
val_func = NNValueFunction(obs_dim, hid1_size)

policy = Policy(obs_dim, act_dim, kl_targ, hid1_size, init_logvar)

episode = 0
n_batch = 0
reward_history = []
while episode < num_episodes:
    trajectories = run_policy(env, policy, episodes=batch_size)

    add_value(trajectories, val_func)  # add estimated values to episodes
    add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
    add_gae(trajectories, gamma, lam)  # calculate advantage
    # concatenate all episodes into single NumPy arrays
    observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
    
    policy.update(observes, actions, advantages)  # update policy
    val_func.fit(observes, disc_sum_rew)  # update value function

    #print avg of batch rewards and store the reward of each episode in the history 
    for i in range(len(trajectories)):        
        reward_history.append(np.sum(trajectories[i]['rewards']))
    reward_history.append(np.sum(trajectories[i]['rewards']))
    
    episode += len(trajectories)    
    n_batch += 1
    
    print ("Batch: "+str(n_batch)+" Reward avg: "+str(avg_batch_rewards(trajectories)))

#plot the reward history
filename = 'plot_rewards.png'
plotLearning(reward_history, filename=filename, window=100)

