# Copyright (c) 2024-2025, Bruno Corcuera
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Find the full license text in the LICENSE file at the root of this repository.

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