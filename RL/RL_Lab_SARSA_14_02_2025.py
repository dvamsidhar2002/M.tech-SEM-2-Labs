#!/usr/bin/env python
# coding: utf-8

# ## D Vamsidhar - 24070149005
# ### SARSA (State-Action-Reward-State-Action)

# <strong>
# 1. Initialization<br>
# 2. The SARSA Update Process - Q(st,at) + alpha[rt + gammaQ(st+1,at+1) - Q(st, at)]<br> 
# 3. Policy Iteration<br>
# </strong>

# In[2]:


# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from envs import Maze
from utils import plot_policy, plot_action_values, test_agent
import warnings

warnings.filterwarnings('ignore')


# In[3]:


env = Maze()


# In[4]:


frame = env.render(mode='rgb_array')
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(frame)


# In[5]:


print(f"Observation space shape: {env.observation_space.nvec}")
print(f"Action space: {env.action_space.n}")


# In[6]:


action_values = np.zeros(shape=(5, 5, 4))


# In[7]:


plot_action_values(action_values)


# In[8]:


def policy(state, epsilon=0.):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av==av.max()))


# In[9]:


action = policy((0,0))
print(f'Action taken in state (0,0): {action}')


# In[10]:


plot_policy(action_values, frame)


# In[11]:


def sarsa(action_values, policy, episodes, alpha=0.1, gamma=0.99, epsilon=0.2):

    for episodes in range(1, episodes + 1):
        state = env.reset()
        action = policy(state, epsilon)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state, epsilon)

            qsa = action_values[state][action]
            next_qsa = action_values[next_state][next_action]
            action_values[state][action] = qsa + alpha * (reward + gamma*next_qsa - qsa)
            state = next_state
            action = next_action


# In[12]:


sarsa(action_values, policy, 1000)


# In[13]:


plot_action_values(action_values)


# In[14]:


plot_policy(action_values, env.render(mode='rgb_array'))


# In[15]:


test_agent(env, policy, episodes=10)


# In[ ]:




