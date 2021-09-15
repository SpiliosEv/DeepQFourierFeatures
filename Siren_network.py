import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from siren_pytorch import SirenNet
import numpy as np  



class DQNNet_siren(nn.Module):
    def __init__(self, input_size, output_size, B, lr=1e-3):
        super(DQNNet_siren, self).__init__()
        self.B = B
        self.B = torch.tensor(B, requires_grad=False)
        
        self.fc1 = nn.Linear (4*input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_size)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)


    def forward(self, x):
        x = x.view(-1,2,1)
        x = torch.matmul(self.B,x)
        x = x.view(-1,4)
        x_cos = torch.cos(2*3.14*x)
        x_sin = torch.sin(2*3.14*x)
        x_hat = torch.cat([x_cos,x_sin], dim=1)
        x_hat = self.fc1(x_hat)
        x_hat = F.relu(x_hat)
        x_hat = self.fc2(x_hat)
        x_hat = F.relu(x_hat)
        x_hat = self.fc3(x_hat)
        
        return x_hat