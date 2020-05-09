# import networkx as nx
import sys
from utils.utils import *
from utils.network import *

if __name__ == '__main__':

    # parameter initialization
    learning_rate = 0.01
    batch_size = 10
    input_size = 24
    epochs = 201

    # object initialization
    dqn = DoubleQNet(input_size, learning_rate, './backup/')
    dqn.train(epochs, batch_size)