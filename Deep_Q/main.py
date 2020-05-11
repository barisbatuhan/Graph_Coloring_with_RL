# import networkx as nx
import sys
from utils.utils import *
from utils.network import *

if __name__ == '__main__':

    # parameter initialization
    learning_rate = 0.01
    batch_size = 10
    input_size = 24
    epochs = 51
    backup_loc = './backup/'
    # object initialization
    dqn = DoubleQNet(input_size, learning_rate, load_path=backup_loc)
    # dqn.train(epochs, batch_size, 20, 100)
    node_cnts = dqn.evaluate('../Matrices/temp/')
    print(node_cnts)