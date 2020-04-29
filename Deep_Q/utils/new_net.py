import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from channel.graph_lib import Graph_Lib


class QNet(nn.Module):

    def __init__(self, input_size, output_size, lr):
        super.__init__(self)
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        fc1 = F.relu(self.fc1(x.view(-1, self.input_size)))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.sigmoid(self.fc3(fc2))
        return fc3

    def adjust_lr(optimizer, decrease_amount, per_epoch):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] -= decrease_amount
            if(param_group['lr'] < 0):
                param_group['lr'] = 0

    def calc_loss(self):
        pass


class DQNet():

    def __init__(self, input_size,
                 lr, lr_dec_amount, lr_dec_per_epoch,
                 eps, eps_dec_amount, eps_dec_per_epoch):
        
        qnet_local = QNet(input_size, 1, lr)
        qnet_target = QNet(input_size, 1, lr)
        pass

    def train(self, batch_size, epochs):
        pass

    def predict(self):
        pass
