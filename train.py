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


def init_gym(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, eps, animate=False):
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

        #take an action from the policy
        if (np.random.uniform(0,1) > eps):
            action = policy.sample(obs)

        else:
            #take a random action, based on the dimension of the action space
            action = np.zeros((1,env.action_space.shape[0]))
            for i in range(env.action_space.shape[0]):
                random_sample = env.action_space.sample()
                action [0][i]= random_sample[i]
        
        actions.append(action)
        obs, reward, done, _ = env.step(action.flatten())
        rewards.append(reward)
        totalReward += reward
    print ("Reward: "+str(totalReward))   
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
        observes, actions, rewards, totalReward = run_episode(env, policy, eps)

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
        gamma: discount factor

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
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


def add_adv(trajectories):
    """ Add advantage .

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        disc_sum_rew = trajectory['disc_sum_rew']
        values = trajectory['values']

        advantages = disc_sum_rew - values
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_adv()

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
    # normalize advantages (standard score), instead of doing mean(disc_sum_rew) - mean(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew

def avg_batch_rewards (trajectories):
    """ Calculate avarage of rewards in the trajectories batch """
    total_reward = []
    for i in range(len(trajectories)):        
        total_reward.append(np.sum(trajectories[i]['rewards']))
    avg = np.mean(total_reward)
    return avg




""" TRAINING START HERE """

env_name = 'MountainCarContinuous-v0'   
#env_name = 'LunarLanderContinuous-v2'   

num_episodes=5000             # OpenAI Gym environment name
gamma = 0.995                 # Discount factor
delta = 0.005                 # D_KL target value
batch_size = 5                # Number of episodes per training batch
hid1_size = 32                # Size of the first hidden layer for value and policy NN
eps = 0.2                     # Epsilon: parameter for eps-greedy choice of action
init_logvar = -1              # Initial policy natural log of variance


env, obs_dim, act_dim = init_gym(env_name)
env = gym.make(env_name)
env.seed(11)
val_func = NNValueFunction(obs_dim, hid1_size)

policy = Policy(obs_dim, act_dim, delta, hid1_size, init_logvar)

episode = 0
n_batch = 0
reward_history = []
while episode < num_episodes:
    trajectories = run_policy(env, policy, episodes=batch_size)

    add_value(trajectories, val_func)  # add estimated values to episodes
    add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
    add_adv(trajectories)  # calculate advantage
    # concatenate all episodes into single NumPy arrays
    observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)

    policy.update(observes, actions, advantages)  # update policy
    val_func.fit(observes, disc_sum_rew)  # update value function

    #print avg of batch rewards and store the reward of each episode in the history 
    for i in range(len(trajectories)):        
        reward_history.append(np.sum(trajectories[i]['rewards']))
    episode += len(trajectories)    
    n_batch += 1
    
    print ("Batch: "+str(n_batch)+" Reward avg: "+str(avg_batch_rewards(trajectories)))
    print ("________________________________")
    #plot the reward history
    filename = env_name+' plot_rewards.png'
    
    plotLearning(reward_history, filename=filename, window=100)



