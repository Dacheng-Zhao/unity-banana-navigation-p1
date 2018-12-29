import torch
import torch.nn as nn
import torch.nn.functional as F


# Create the architect of the neural network, the brain

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

# Forward propogate        
    def forward(self, state):
        # Activation functions relu
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class Dueling_DQN(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        super(Dueling_DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_adv = nn.Linear(state_size, fc1_units)
        self.fc1_val = nn.Linear(state_size, fc1_units)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.fc3_adv = nn.Linear(fc2_units, action_size)
        self.fc3_val = nn.Linear(fc2_units, 1)

# Forward propogate
    def forward(self, state):
        x_adv = F.relu(self.fc1_adv(state))
        x_adv = F.relu(self.fc2_adv(x_adv))
        adv = self.fc3_adv(x_adv)
        
        x_val = F.relu(self.fc1_val(state))
        x_val = F.relu(self.fc2_val(x_val))
        val = self.fc3_val(x_val).expand(x_val.size(0), self.action_size)

        return val + adv - adv.mean(1).unsqueeze(1).expand(x_val.size(0), self.action_size)

# add more nodes for drop
class DQN_Drop(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        super(DQN_Drop, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
# Forward propogate        
    def forward(self, state):
        # Activation functions relu
        x = F.relu(self.fc1(state))
        torch.nn.Dropout(0.2)
        x = F.relu(self.fc2(x))
        torch.nn.Dropout(0.2)
        q_values = self.fc3(x)
        return q_values
