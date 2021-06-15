'''
This is the NN Model that we will use with the DQN
This model contains the following
fc1 => nn.linear
fc2 => nn.linear
fc3 => nn.linear

The model also makes use of dropouts with p=0.5
'''

import torch
import torch.nn as nn
import torch.nn.functional as functional

class QNetwork(nn.Module):
    """ Map States to Actions """

    def __init__(self, stateSize, actionSize):
        """ Add the first layer, add a hidden layers, add last year and dropout """
        super(QNetwork, self).__init__()

        dropoutProb = 0.5
        fc = [128, 128]

        self.fc1 = nn.Linear(stateSize, fc[0])
        self.fc2 = nn.Linear(fc[0], fc[1])
        self.fc3 = nn.Linear(fc[1], actionSize)
        
        self.dropout = nn.Dropout(p=dropoutProb)


    def forward(self, state):
        x = functional.relu(self.fc1(state))
        x = functional.relu(self.fc2(x))
        return self.fc3(x)