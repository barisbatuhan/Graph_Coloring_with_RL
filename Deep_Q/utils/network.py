import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from channel.graph_lib import Graph_Lib

class QNet(nn.Module):

    def __init__(self, input_dims, lr):
        super(QNet, self).__init__()
        # gpu or cpu set
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        # network parameters
        self.hidden_layer_neurons = 512
        # layers
        self.fc1 = nn.Linear(input_dims, self.hidden_layer_neurons)
        self.fc2 = nn.Linear(self.hidden_layer_neurons, 1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        action = self.fc2(fc1)
        return action

class DoubleQNet():

    def __init__(self, input_dims, learning_rate, discount_rate, discount_count):
        # learning parameters
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.discount_count = discount_count
        self.step_counter = 0
        # networks
        self.local_qnet = QNet(input_dims, learning_rate)
        self.target_qnet = QNet(input_dims, learning_rate)
        # network parameters
        # communication with C++ codes

    def train(self, epochs, batch_size):
        init_lr = self.learning_rate
        total_graph_index = 0

        for epoch in range(epochs):
            batch = 64
            n, e = 50, 100
            for cnt in range(batch):
                graphs = Graph_Lib()
                graphs.insert_batch(batch, n, e)
                num_nodes = n
                graphs.init_node_embeddings()
                graphs.init_graph_embeddings()
                colored_arrs = [[False]*num_nodes]*batch
                for colored_cnt in range(num_nodes):
                    nodes_to_color = decide_node_coloring(graphs, num_nodes, batch)
                    graphs.color_batch(nodes_to_color)
                    # update network
                    if(colored_cnt % 3 == 0):
                        graph_embeds = graphs.update_graph_embeddings()
                    pass

                graph.reset_batch()

    def decide_node_coloring(self, graphs, num_nodes, batch):
        for el in range(batch):

        pass
