import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from channel.graph_lib import Graph_Lib

class QNet(nn.Module):

    def __init__(self, input_dims, lr):
        super(QNet, self).__init__()
        # gpu or cpu set
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
class DoubleQNet():

    def __init__(self, input_dims, learning_rate, load_path=None):
        # learning parameters
        self.learning_rate = learning_rate
        self.step_counter = 0
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.999
        # networks
        self.local_qnet = QNet(input_dims, learning_rate)
        self.target_qnet = QNet(input_dims, learning_rate)

        if(load_path != None):
            self.local_qnet.load_model(load_path + 'local_qnet.params')
            self.target_qnet.load_model(load_path + 'target_qnet.params')

    
    def evaluate(self, path):
        graphs = Graph_Lib()
        node_cnts = graphs.read_batch(path)
        batch = len(node_cnts)
        graphs.init_node_embeddings()
        graphs.init_graph_embeddings()    
        colored_arrs = []
        max_node = -1
        avg_node = 0
        for cnt in node_cnts:
            avg_node += cnt
            if(cnt > max_node):
                max_node = cnt
            colored_arrs.append([False]*cnt)
        max_colors = [-1]*batch
        avg_node /= batch
        for colored_cnt in range(max_node):
            actions, _, _ = self.decide_node_coloring(graphs, node_cnts, batch, colored_arrs, colored_cnt)
            colors = graphs.color_batch(actions)
            _ = self.get_rewards(batch, colors, max_colors)
            if(colored_cnt % 3 == 0):
                graphs.update_graph_embeddings()
        return max_colors
    
    def train(self, epochs, batch_size, min_nodes, max_nodes):
        batch = batch_size
        for epoch in range(1, epochs):
            loss_total = 0
            graphs = Graph_Lib()
            node_cnts = graphs.insert_batch(batch, min_nodes, max_nodes)
            graphs.init_node_embeddings()
            graphs.init_graph_embeddings()
            
            colored_arrs = []
            max_node = -1
            avg_node = 0
            for cnt in node_cnts:
                avg_node += cnt
                if(cnt > max_node):
                    max_node = cnt
                colored_arrs.append([False]*cnt)
            # colored_arrs = np.full((batch, num_nodes), False, dtype=bool)
            max_colors = [-1]*batch
            avg_node /= batch
            
            for colored_cnt in range(max_node):
                actions, q_pred, not_finished = self.decide_node_coloring(graphs, node_cnts, batch, colored_arrs, colored_cnt)
                colors = graphs.color_batch(actions)
                rewards = self.get_rewards(batch, colors, max_colors)
                q_target = self.get_loss(graphs, batch, rewards, actions)
                # print(np.linalg.norm(q_target.detach().numpy() - q_pred.detach().numpy()))
                loss = self.local_qnet.loss(q_target, q_pred).to(self.local_qnet.device)
                loss.backward()
                loss_total += loss / not_finished
                self.local_qnet.optimizer.step()
                self.step_counter += 1
                if(colored_cnt % 3 == 0):
                   graphs.update_graph_embeddings()
                pass
            loss_total /= max_node
            print("Epoch:", epoch, "--- Loss:", loss_total) 
            graphs.reset_batch()
            if(self.epsilon > 0.05): # decay epsilon
                    self.epsilon *= self.epsilon_decay
            
            if(epoch % 50 == 0): # set target as local
                # print("Epoch:", epoch, "--- Loss:", loss_total)    
                self.local_qnet.save_model('./backup/local_qnet.params')
                self.target_qnet.save_model('./backup/target_qnet.params')
                self.target_qnet.load_state_dict(self.local_qnet.state_dict())

            
    def decide_node_coloring(self, graphs, node_cnt, batch, colored_arrs, colored_cnt):
        actions = []
        q_pred = []
        not_finished = batch
        for el in range(batch):
            max_action = -9999
            max_node = -1
            node_embeds = graphs.get_node_embed(el)
            graph_embeds = graphs.get_graph_embed(el)
            if(colored_cnt >= node_cnt[el]):
                not_finished -= 1
                actions.append(-1)
                continue
            elif random.random() > self.epsilon:
                for node in range(node_cnt[el]):
                    if colored_arrs[el][node]:
                        continue
                    embeddings = np.concatenate([node_embeds[node], graph_embeds])
                    embeddings = torch.from_numpy(embeddings).float()
                    with torch.no_grad():
                        action = self.local_qnet(embeddings)
                    if(max_action < action):
                        max_node = node
                        max_action = action
                colored_arrs[el][max_node] = True
                actions.append(max_node)
                q_pred.append(max_action)

            else:
                found = False
                while not found:
                    node = random.randint(0, node_cnt[el] - 1)
                    if not colored_arrs[el][node]:
                        found = True
                        colored_arrs[el][node] = True
                        embeddings = np.concatenate([node_embeds[node], graph_embeds])
                        embeddings = torch.from_numpy(embeddings).float()
                        with torch.no_grad():
                            action_val = self.local_qnet(embeddings)
                        q_pred.append(action_val)
                        actions.append(node)

        return actions, torch.Tensor(q_pred).requires_grad_(), not_finished
    
    def get_rewards(self, batch, colors, max_colors):
        rewards = [0]*batch
        for el in range(batch):
            # print(max_colors[el], colors[el])
            if(colors[el] == -1):
                rewards[el] = -9999
            else:
                rewards[el] = - max(0, - max_colors[el] + colors[el])
                if(max_colors[el] < colors[el]):
                    max_colors[el] = colors[el]
        return np.array(rewards)

    def get_loss(self, graphs, batch, rewards, actions):
        losses = []
        for el in range(batch):
            node_embeds = graphs.get_node_embed(el)
            graph_embeds = graphs.get_graph_embed(el)
            embeddings = np.concatenate([node_embeds[actions[el]], graph_embeds])
            embeddings = torch.from_numpy(embeddings).float()
            if(rewards[el] == -9999):
                continue
            with torch.no_grad():
                losses.append(rewards[el] + self.gamma * self.target_qnet(embeddings))
        return torch.Tensor(losses).requires_grad_()
