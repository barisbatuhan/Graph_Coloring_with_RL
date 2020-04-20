# import networkx as nx
import sys
from utils.utils import *
from utils.network import *

if __name__ == '__main__':
    
    # parameter initialization
    train_path = './../Matrices/small/'
    learning_rate = 0.01
    discount_rate = 0.001
    discount_count = 10
    batch_size = 32
    input_size = 25
    epochs = 1
    
    # object initialization
    dqn = DoubleQNet(input_size, learning_rate, discount_rate, discount_count)
    
    #real process
    files = get_matrix_names(train_path, True)
    dqn.train(epochs, batch_size, files)