#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


class K_arm_bandit: 
    '''
    epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    iterations: number of steps (int)
    eps: probability of random action 0 < eps < 1 (float)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, iterations, eps, mu = 'random'):
        '''
        k = Number of action
        iter = Number of steps the agent takes in each episode
        eps = Search probablity
        mu = User-defined q_star array. If mu = 'random' then q_star is a random array
        q_star = Mean expected reward for each action
        total_steps = Total steps taken by the agent so far
        k_steps = Step count for each action
        total_reward = Net mean total reward achieved by the agent
        k_reward = Mean total reward for each action
        reward_track = Track of mean total reward acheived by the agent after every step
        '''
        self.k = k
        self.iter = iterations    
        self.eps = eps
        if type(mu) == list or type(mu).__module__ == np.__name__:           
            self.q_star = np.array(mu)
        elif mu == 'random':
            self.q_star = np.random.normal(0, 1, k)
            
        self.total_steps = 0
        self.k_steps = np.zeros(k)
        self.total_reward = 0
        self.k_reward = np.zeros(k)
        self.reward_track = np.zeros(iterations)
    
    def eps_greedy(self):
        
        # Greedily select an action
        action = np.argwhere(self.k_reward == np.max(self.k_reward)).reshape(-1)
        action = np.random.choice(action)
        
        # Randomly select an action
        if(self.eps > 0):
            choice = np.random.choice([0, 1], p = [1-self.eps, self.eps])
            if(choice == 1):
                action = np.random.choice(self.k)
                        
        reward = np.random.normal(self.q_star[action], 1)
        self.k_steps[action] = self.k_steps[action] + 1
        self.total_steps = self.total_steps + 1
        self.k_reward[action] = self.k_reward[action] + (reward - self.k_reward[action])/self.k_steps[action]
        self.total_reward = self.total_reward + (reward - self.total_reward)/self.total_steps
    
    def run_greedy(self):
        for i in range(self.iter):
            self.eps_greedy()
            self.reward_track[i] = self.total_reward


# In[6]:


k = 10
iterations = 1000
episodes = 50

bandit1_reward = np.zeros(iterations)
bandit2_reward = np.zeros(iterations)
bandit3_reward = np.zeros(iterations)

for i in range(episodes):
    bandit1 = K_arm_bandit(10, 1000, 0)
    bandit2 = K_arm_bandit(10, 1000, 0.01, bandit1.q_star)
    bandit3 = K_arm_bandit(10, 1000, 0.1, bandit1.q_star)
    
    bandit1.run_greedy()
    bandit2.run_greedy()
    bandit3.run_greedy()
    
    bandit1_reward = bandit1_reward + (bandit1.reward_track - bandit1_reward)/(i+1)
    bandit2_reward = bandit2_reward + (bandit2.reward_track - bandit2_reward)/(i+1)
    bandit3_reward = bandit3_reward + (bandit3.reward_track - bandit3_reward)/(i+1)

plt.figure(figsize=(12,8))
plt.plot(bandit1_reward, label="$\epsilon=0$ (greedy)")
plt.plot(bandit2_reward, label="$\epsilon=0.01$ (greedy)")
plt.plot(bandit3_reward, label="$\epsilon=0.1$ (greedy)")
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon-greedy$ Rewards after " + str(episodes) 
    + " Episodes")
plt.show()
