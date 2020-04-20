import os
import numpy as np

def get_matrix_names(path, include_path=False):
    files = []
    names = os.listdir(path)
    for name in names:
        if '.mtx' in name:
            if include_path:
                files.append(path + name)
            else:
                files.append(name)
    return files

