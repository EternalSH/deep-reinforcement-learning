import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from memory import ReplayBuffer
from noise import OUNoise

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, config, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.config = config
        self.device = config['device']

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.noise_epsilon = config['NOISE_EPSILON']
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config['LR_ACTOR'])
        self.hard_update(self.actor_local, self.actor_target)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config['LR_CRITIC'], weight_decay=config['WEIGHT_DECAY'])
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise((1, action_size), random_seed, 0.0, config['OU_THETA'], config['OU_SIGMA'])
        self.noise_epsilon = config['NOISE_EPSILON']

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.config, random_seed)
    
    def step(self, t, state, action, reward, next_state, done, agent_index):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)
        
        if t % self.config['DDPG_UPDATE_EVERY'] == 0 and len(self.memory) > self.config['BATCH_SIZE']:
            for _ in range(self.config['DDPG_LEARN_TIMES']):
                experiences = self.memory.sample()
                self.learn(experiences, agent_index)            

    def act(self, states):
        states = torch.from_numpy(states).float().to(self.device)
        actions = np.zeros((1, self.action_size))
        
        self.actor_local.eval()
        
        with torch.no_grad():
            for agent_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_num, :] = action
                
        self.actor_local.train()        
        actions += self.noise_epsilon * self.noise.sample()
            
        return np.clip(actions, -1, 1)
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, agent_index):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        gamma = self.config['GAMMA']
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        if agent_index == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)        
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        if agent_index == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)   

        # ---------------------------- update noise ---------------------------- #
        self.noise_epsilon = max(self.noise_epsilon - self.config['NOISE_EPSILON_DECAY'], self.config['NOISE_EPSILON_MIN'])
        self.noise.reset()

    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        tau = self.config['TAU']
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)