# import networkx as nx
import sys
from network import *

if __name__ == '__main__':
    
    # PARAMETER INITIALIZATION
    learning_rate = 0.005
    batch_size = 3
    input_size = 24
    epochs = 120
    min_nodes = 100
    max_nodes = 500

    # TRAINING 
    dqn = DoubleQNet(input_size, learning_rate)
    dqn.train(epochs, batch_size, min_nodes, max_nodes)
    
    # EVALUATION
    backup_loc = './backup/' # backup location where the model saved will be loaded
    eval_path = '../Matrices/small/' # folder path matrices for evaluation (will be taken as single batch)
    dqn = DoubleQNet(input_size, learning_rate, backup_loc)
    dqn.evaluate(eval_path)