import ctypes
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class Graph_Lib(object):
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary('./source/lib_graph.so')
        # batch embedding functions -saklandi
        # self.lib.initialize_graph_embeddings_for_batch.argtypes = []
        # self.lib.initialize_graph_embeddings_for_batch.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        # graph reader
        self.lib.insert_batch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.insert_batch.restype = ctypes.POINTER(ctypes.c_int)
        self.lib.reset_batch.argtypes = []
        self.lib.read_batch.argtypes = [ctypes.c_char_p]
        self.lib.read_batch.restype = ctypes.POINTER(ctypes.c_int)
        # embedding init
        self.lib.init_node_embeddings.argtypes = []
        self.lib.init_graph_embeddings.argtypes = []
        # embedding update
        self.lib.update_graph_embeddings.argtypes = []
        # getters
        self.lib.get_node_embed.argtypes = [ctypes.c_int, ctypes.POINTER(
            ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.lib.get_node_embed.restype = ctypes.POINTER(
            ctypes.POINTER(ctypes.c_float))
        self.lib.get_graph_embed.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.get_graph_embed.restype = ctypes.POINTER(ctypes.c_float)
        # coloring of batch
        self.lib.color_batch.argtypes = [ctypes.POINTER(
            ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.lib.color_batch.restype = ctypes.POINTER(ctypes.c_int)
        
        # self.lib.update_node_embeddings.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

    def insert_batch(self, batch, min_n, max_n):
        nodes = self.lib.insert_batch(batch, min_n, max_n)
        return nodes[:batch]

    def reset_batch(self):
        self.lib.reset_batch()

    def read_batch(self, path):
        a = ctypes.c_int(0)
        location = ctypes.c_char_p(path.encode('utf-8'))
        nodes = self.lib.read_batch(location, ctypes.byref(a))
        return nodes[:a.value]

    def init_node_embeddings(self):
        self.lib.init_node_embeddings()

    def init_graph_embeddings(self):
        self.lib.init_graph_embeddings()

    def update_graph_embeddings(self):
        self.lib.update_graph_embeddings()

    def get_node_embed(self, index):
        a = ctypes.c_int(0)
        b = ctypes.c_int(0)
        res = self.lib.get_node_embed(index, ctypes.byref(a), ctypes.byref(b))
        arr = []
        for r in range(a.value):
            temp = []
            for c in range(b.value):
                val = res[r][c]
                temp.append(val)
            arr.append(temp)
        return np.array(arr)

    def get_graph_embed(self, index):
        a = ctypes.c_int(0)
        res = self.lib.get_graph_embed(index, ctypes.byref(a))
        return np.array(res[:a.value])

    def color_batch(self, nodes):
        a = ctypes.c_int(0)
        start = (ctypes.c_int * len(nodes))(*nodes)
        res = self.lib.color_batch(start, ctypes.byref(a))
        return np.array(res[0:a.value])

    # def initialize_graph_embeddings_for_batch(self):
    #     a = ctypes.c_int(0)
    #     b = ctypes.c_int(0)
    #     res = self.lib.initialize_graph_embeddings_for_batch(ctypes.byref(a), ctypes.byref(b))
    #     return (res, a.value, b.value)
