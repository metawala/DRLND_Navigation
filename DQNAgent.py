import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as functional
import torch.optim as optim

"""
DEFINE CONSTANTS
"""
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ Interacts with and learns from the environment """
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize

        # Q-Network with ADAM Optimizer
        self.qnetworkLocal = QNetwork(stateSize, actionSize).to(device)
        self.qnetworkTarget = QNetwork(stateSize, actionSize).to(device)
        self.optimizer = optim.Adam(self.qnetworkLocal.parameters(), lr=LR)

        # Replay memory and Initialize timestep for updating at UPDATE_EVERY
        self.memory = ReplayBuffer(actionSize, BUFFER_SIZE, BATCH_SIZE)
        self.tStep = 0
    
    def step(self, state, action, reward, nextState, done):
        # Save memory and Learn every UPDATE_EVERY step
        self.memory.add(state, action, reward, nextState, done)
        self.tStep = (self.tStep + 1) % UPDATE_EVERY

        if self.tStep == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps = 0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetworkLocal.eval()
        
        with torch.no_grad():
            actionValues = self.qnetworkLocal(state)
        self.qnetworkLocal.train()

        # Epsilon Greedy action implementation
        if random.random() > eps:
            return np.argmax(actionValues.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.actionSize))

    def learn(self, experiences, gamma):
        states, actions, rewards, nextStates, dones = experiences
        # 1. Get max predicted Q values from target model
        # 2. Get Q Targets from current state
        # 3. Get Expected Q values from local model
        QTargetsNext = self.qnetworkTarget(nextStates).detach().max(1)[0].unsqueeze(1)
        QTargets = rewards + (gamma * QTargetsNext * (1 - dones))
        QExpected = self.qnetworkLocal(states).gather(1, actions)

        # Compute Loss, minimize loss - zero_grad()
        loss = functional.mse_loss(QExpected, QTargets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.softUpdate(self.qnetworkLocal, self.qnetworkTarget, TAU)

    def softUpdate(self, localModel, targetModel, tau):
        for targetParam, localParam in zip(targetModel.parameters(), localModel.parameters()):
            targetParam.data.copy_(tau*localParam.data + (1.0 - tau)*targetParam.data)

class ReplayBuffer:

    def __init__(self, actionSize, bufferSize, batchSize):
        self.actionSize = actionSize
        self.memory = deque(maxlen = bufferSize)
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "nextState", "done"])

    def add(self, state, action, reward, nextState, done):
        exp = self.experience(state, action, reward, nextState, done)
        self.memory.append(exp)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batchSize)

        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        nextStates = torch.from_numpy(np.vstack([exp.nextState for exp in experiences if exp is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, nextStates, dones)

    def __len__(self):
        return len(self.memory)