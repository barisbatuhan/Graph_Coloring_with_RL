import ctypes
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))  

class Graph_Lib(object):
    def __init__(self):   
        self.lib = ctypes.cdll.LoadLibrary('./source/lib_graph.so')
        self.lib.print_graph_features.argtypes = []
        # graph constructors
        self.lib.insert_graph.argtypes = [ctypes.c_int,ctypes.c_int]
        self.lib.read_graph.argtypes = [ctypes.c_char_p]
        self.lib.read_graph.restype = ctypes.c_int
        # single graph embeddings
        self.lib.init_node_embeddings.argtypes = []
        self.lib.init_node_embeddings.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        self.lib.init_graph_embeddings.argtypes = []
        self.lib.init_graph_embeddings.restype = ctypes.POINTER(ctypes.c_float)
        # resetting function for heap memory
        self.lib.reset.argtypes = []
        # batch embedding functions
        self.lib.initialize_graph_embeddings_for_batch.argtypes = []
        self.lib.initialize_graph_embeddings_for_batch.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

    def print_graphs(self):
        self.lib.print_graph_features()

    def insert_graph(self, n, e):
        return self.lib.insert_graph(n,e)

    def initialize_graph_embeddings_for_batch(self):
        a = ctypes.c_int(0)
        b = ctypes.c_int(0)
        res = self.lib.initialize_graph_embeddings_for_batch(ctypes.byref(a), ctypes.byref(b))
        return (res, a.value, b.value)

    # NEWLY ADDED
    def read_graph(self, graph_name):
        gname = graph_name.encode('utf-8')
        num_nodes = self.lib.read_graph(gname)
        return num_nodes
    
    def init_node_embeddings(self):
        res = self.lib.init_node_embeddings()
        return res
    
    def init_graph_embeddings(self):
        res = self.lib.init_graph_embeddings()
        return res
    
    def reset(self):
        self.lib.reset()