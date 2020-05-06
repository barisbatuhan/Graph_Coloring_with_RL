# import networkx as nx
# import sys
# from utils.utils import *
# from utils.network import *
#
#
#
# if __name__ == '__main__':
#
#     # parameter initialization
#     train_path = './../Matrices/small/'
#     learning_rate = 0.01
#     discount_rate = 0.001
#     discount_count = 10
#     batch_size = 1
#     input_size = 24
#     epochs = 1
#
#     # object initialization
#     dqn = DoubleQNet(input_size, learning_rate, discount_rate, discount_count)
#
#     #real process
#     files = get_matrix_names(train_path, True)
#     dqn.train(epochs, batch_size, files)

from channel.graph_lib import Graph_Lib

g = Graph_Lib()

g.insert_batch(32, 5, 10)
print("INITIALIZING NODE EMBEDS...")
g.init_node_embeddings()
print("INITIALIZING FINISHED...")
g.init_graph_embeddings()
print("GETTING NODE EMBEDS...")
ne, r, c = g.get_node_embed(17)
print("GETTING FINISHED...")
ge, s = g.get_graph_embed(31)

nodes_to_color = [4]*32
print("COLORING...")
g.color_batch(nodes_to_color)
print("UPDATING GRAPH EMBEDS...")
g.update_graph_embeddings()
print("GETTING NODE EMBEDS...")
ne, r, c = g.get_node_embed(17)
print("GETTING FINISHED...")
ge, s = g.get_graph_embed(31)

# print("PRINTING NODE EMBEDS")
#
# for i in range(r):
#     for j in range(c):
#         print(ne[i][j])
#
print("PRINTING GRAPH EMBEDS...")

for k in range(s):
    print(k,"-",ge[k])
