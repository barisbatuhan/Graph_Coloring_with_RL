import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from channel.graph_lib import Graph_Lib

class QNet(nn.Module):
    
    def __init__(self, input_dims):
        super(QNet, self).__init__()
        # gpu or cpu set
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        # network parameters
        self.hidden_layer_neurons = 512
        # layers
        self.fc1 = nn.Linear(input_dims, self.hidden_layer_neurons)
        self.fc2 = nn.Linear(self.hidden_layer_neurons, 1)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        action = F.sigmoid(self.fc2(fc1))
        return action

class DoubleQNet():
    
    def __init__(self, input_dims, learning_rate, discount_rate, discount_count):
        # learning parameters
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.discount_count = discount_count
        self.step_counter = 0
        # networks
        self.local_qnet = QNet(input_dims)
        self.target_qnet = QNet(input_dims)
        # network parameters
        self.optimizer = optim.RMSprop(self.local_qnet.parameters(), lr=self.learning_rate)
        # communication with C++ codes
        self.graph = Graph_Lib()

    def train(self, epochs, batch_size, graph_names): 
        init_lr = self.learning_rate      
        total_graph_index = 0
        
        for epoch in range(epochs):   
            for batch in range(batch_size):
                # reading the graph
                graph_name = graph_names[total_graph_index]
                num_nodes = self.graph.read_graph(graph_name)
                colored_arr = [False]*num_nodes # node count will be placed here
                # reading node embeddings
                node_embeds = self.graph.init_node_embeddings(num_nodes)
                graph_embeds = self.graph.init_graph_embeddings()
                graph_embeds = graph_embeds[0:16]
                colored_cnt = 0
                while colored_cnt < num_nodes:
                    # TO DO: ADD ALL THE PROCESSES FOR CHOOSING THE NODE TO COLOR
                    node_to_color = 0 # TO DO: CHANGE IT AND SELECT THE NODE CORRECTLY
                    selected_color = self.graph.color_node(node_to_color)
                    colored_cnt +=1
                    colored_arr[node_to_color] = True
                    node_embeds = self.graph.update_node_embeddings(node_to_color, selected_color, num_nodes)
                    # graph_embeds = self.graph.update_graph_embeddings()
                    if(colored_cnt % 3 == 0):
                        graph_embeds = self.graph.update_graph_embeddings()
                    # TO DO: UPDATE LOCAL QNET
                    break
                total_graph_index += 1
                if(total_graph_index % len(graph_names) == 0):
                    total_graph_index = 0
                
                self.graph.reset()
            # TO DO: UPDATE TARGET NETWORK

    def get_max_action(colored_info_arr):
        max_action = -1
        max_val = -1
        for node in range(colored_info_arr):
            if(colored_info_arr[node] == True): # already colored node
                continue
            val = self.target_qnet.eval() + self.local_qnet.eval()
            if(val > max_val):
                max_val = val
                max_action = node
        return max_action