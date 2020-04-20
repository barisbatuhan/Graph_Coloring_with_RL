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
        # batch embedding functions
        self.lib.initialize_graph_embeddings_for_batch.argtypes = []
        self.lib.initialize_graph_embeddings_for_batch.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

        # graph reader
        self.lib.read_graph.argtypes = [ctypes.c_char_p]
        self.lib.read_graph.restype = ctypes.c_int
        # single graph - embedding initialization
        self.lib.init_node_embeddings.argtypes = []
        self.lib.init_node_embeddings.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        self.lib.init_graph_embeddings.argtypes = []
        self.lib.init_graph_embeddings.restype = ctypes.POINTER(ctypes.c_float)
        # coloring of single node
        self.lib.color_node.argtypes = [ctypes.c_int]
        self.lib.color_node.restype = ctypes.c_int
        # updating embeddings
        self.lib.update_node_embeddings.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.update_node_embeddings.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        self.lib.update_graph_embeddings.argtypes = []
        self.lib.update_graph_embeddings.restype = ctypes.POINTER(ctypes.c_float)
        # resetting function for heap memory
        self.lib.reset.argtypes = []

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
    
    def init_node_embeddings(self, num_nodes):
        embeds = self.lib.init_node_embeddings()
        embeds = embeds[0:9]
        for i in range(9):
            embeds[i] = embeds[i][0:num_nodes]
        return embeds
    
    def init_graph_embeddings(self):
        embeds = self.lib.init_graph_embeddings()
        return embeds[0:16]
    
    def color_node(self, node):
        return self.lib.color_node(node)

    def update_node_embeddings(self, node, color, num_nodes):
        embeds = self.lib.update_node_embeddings(node, color)
        embeds = embeds[0:9]
        for i in range(9):
            embeds[i] = embeds[i][0:num_nodes]
        return embeds

    def update_graph_embeddings(self):
        embeds = self.lib.update_graph_embeddings()
        return embeds[0:16]
    
    def reset(self):
        self.lib.reset()