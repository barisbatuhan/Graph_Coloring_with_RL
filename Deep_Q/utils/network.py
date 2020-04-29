import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from channel.graph_lib import Graph_Lib


class QNet(nn.Module):

    def __init__(self, input_dims, seed=42):
        super(QNet, self).__init__()
        # gpu or cpu set
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        # network parameters
        self.hidden_layer_neurons = 512
        # self.seed = T.manmual_seed(seed)
        # layers
        self.fc1 = nn.Linear(input_dims, self.hidden_layer_neurons)
        self.fc2 = nn.Linear(self.hidden_layer_neurons, 1)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        action = F.sigmoid(self.fc2(fc1))
        return action
    
    def loss(self, qval, next_qval, lr, reward, epsilon):
        return qval + lr * (reward + (epsilon * next_qval) - qval)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class DoubleQNet():

    def __init__(self, input_dims,
                 learning_rate,
                 discount_rate,
                 discount_count,
                 epsilon = 0.1,
                 epsilon_dec = 0.0001,
                 epsilon_dec_count = 1000,
                 epsilon_min = 0,
                 save_per_epochs=100,
                 steps_for_graph_embed_update=3):
        
        # learning parameters
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.discount_count = discount_count
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_dec_count = epsilon_dec_count
        self.epsilon_min = epsilon_min
        self.step_counter = 0
        self.save_per_epochs = save_per_epochs
        self.steps_for_graph_update = steps_for_graph_embed_update
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
            print("- In epoch:", epoch + 1, "/", epochs)

            if(epoch != 0 and epoch % self.save_per_epochs == 0):
                self.save_models()

            for batch in range(batch_size):
                print("---- In batch:", batch + 1, "/", batch_size)
                # reading the graph
                graph_name = graph_names[total_graph_index]
                num_nodes = self.graph.read_graph(graph_name)
                # node count will be placed here
                colored_arr = [False]*num_nodes
                # initializing embeddings and colored node count
                node_embeds = self.graph.init_node_embeddings(num_nodes)
                graph_embeds = self.graph.init_graph_embeddings()
                colored_cnt = 0
                max_color = -1
                state, action, local_q_val, target_q_val = self.choose_action(colored_arr, num_nodes, node_embeds, graph_embeds)
                n_state, n_action, n_local_q_val, n_target_q_val = None
                
                while colored_cnt < num_nodes: #colors all the nodes
                    selected_color = self.graph.color_node(node_to_color)
                    colored_cnt += 1
                    colored_arr[node_to_color] = True
                    node_embeds = self.graph.update_node_embeddings(node_to_color, selected_color, num_nodes)
                    if(colored_cnt % self.steps_for_graph_update == 0):
                        graph_embeds = self.graph.update_graph_embeddings(node_to_color)
                    
                    n_state, n_action, n_local_q_val, n_target_q_val = self.choose_action(colored_arr, num_nodes, node_embeds, graph_embeds)

                    reward = 0
                    if(selected_color > max_color): # setting loss as increase in colors
                        reward = -1
                        max_color = selected_color
                    
                    if np.random.random() < 0.5: # choose to update local network
                        # loss= local_q_val+ learning_rate*(reward+ (discount*n_target_q_val)-local_q_val).to(qlocal.device)
                        self.local_qnet.train()
                        self.local_qnet.loss(local_q_val, n_target_q_val, self.learning_rate, reward, self.epsilon).to(self.local_qnet.device)
                        loss.backward()
                    
                    else: # choose to update local network
                        self.target_qnet.train()
                        self.target_qnet.loss(target_q_val, n_local_q_val, self.learning_rate, reward, self.epsilon).to(self.target_qnet.device)
                        loss.backward()

                    state = n_state
                    action = n_action
                    local_q_val = n_local_q_val
                    target_q_val = n_target_q_val
                    
                    if(self.step_counter % self.discount_count == 0): # learning rate is lowered in time
                        self.learning_rate -= self.discount_rate
                        if(self.learning_rate < 0):
                            self.learning_rate = 0
                
                # checks if total number of graphs is less then batch size
                # if so, when all graphs end, the graph counter resets
                total_graph_index += 1
                if(total_graph_index % len(graph_names) == 0):
                    total_graph_index = 0

                self.graph.reset() # the graph info is reset in c++, since new graph will be processed next

    def choose_action(self, colored_info_arr, num_nodes, node_embeds, graph_embeds):
        self.step_counter += 1
        if(self.step_counter % self.epsilon_dec_count == 0): #decreasing random selecion probability in time
            self.epsilon -= self.epsilon_dec
            if(self.epsilon <= self.epsilon_min):
                self.epsilon = self.epsilon_min

        if np.random.random() < self.epsilon:  # random choice
            node_to_color = np.random.randint(0, num_nodes - 1)
            while colored_info_arr[node_to_color] == True:
                node_to_color = np.random.randint(0, num_nodes - 1)
            
            state = [0]*(len(node_embeds) + len(graph_embeds))
            index = 0
            for ge in graph_embeds:
                state[index] = ge
                index += 1
            for i in range(len(node_embeds)):
                state[index] = node_embeds[i][node_to_color]
                index += 1
            
            l_value, t_value = None
            with T.no_grad():
                self.local_qnet.eval()
                self.target_qnet.eval()
                l_value = self.local_qnet.forward(state)
                t_value = self.target_qnet.forward(state)
            return node_to_color, state, l_value, t_value
        
        else:  # max action choice
            max_action = -9999
            node_to_color = -1
            self.local_qnet.eval()
            self.target_qnet.eval()
            for node in range(num_nodes):
                if(colored_info_arr[node] == True):  # already colored node
                    continue
                # create the state
                state = [0]*(len(node_embeds) + len(graph_embeds))
                index = 0
                for ge in graph_embeds:
                    state[index] = ge
                    index += 1
                for i in range(len(node_embeds)):
                    state[index] = node_embeds[i][node]
                    index += 1
          
                with T.no_grad():
                    l_value = self.local_qnet.forward(state)
                    t_value = self.target_qnet.forward(state)
                    value = max(l_value, t_value)
                    if(value > max_action):
                        max_action = value
                        node_to_color = node
   
            return node_to_color, state, l_value, t_value

    def save_models(self):
        self.local_qnet.save_checkpoint()
        self.target_qnet.save_checkpoint()

    def load_models(self):
        self.local_qnet.load_checkpoint()
        self.target_qnet.load_checkpoint()
