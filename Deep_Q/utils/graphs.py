import numpy as np
from channel.graph_lib import Graph_Lib
from utils.new_net import QNet

class Graphs:

    def __init__(batch_size, filepaths):
        self.files = filepaths
        self.batch_size = batch_size
        self.current = 0
        self.graphs = [[]]
    
    def reset(self):
        for g in self.graphs:
            g.reset()
        self.graphs = [[]]

    def next(self):
        self.reset()
        for index in range(self.current, self.current + self.batch_size):
            i = index % len(self.files)
            g = Graph_Lib()
            num_nodes = g.read_graph(self.files[i])
            node_embeds = g.init_node_embeddings(num_nodes)
            graph_embeds = g.init_graph_embeddings()
            colored_info = [False]*num_nodes
            self.graphs.append([g, num_nodes, colored_info, node_embeds, graph_embeds])
    
    def get_batch_states(self, qnet, ):
        states = np.array(shape=(batch_size))
        for g in self.graphs:
            for node in range(g[1]):
                if(g[3][node] == True):
                    continue   
                with qnet.no_grad():







