import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super(Controller, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_dim, action_dim),
        )

    def forward(self, x):
        raw_actions = self.model(x)
        
        steering = torch.tanh(raw_actions[:, 0:1])        # [-1,1]
        gas = torch.sigmoid(raw_actions[:, 1:2])          # [0,1]   
        brake =  torch.sigmoid(raw_actions[:, 2:3]) * .8  # [0,0.8] 
        
        actions = torch.cat([steering, gas, brake], dim=1)
    
        return actions

    def get_action(self, state):
        with torch.no_grad():  
            action = self.forward(state)
        return action.squeeze()