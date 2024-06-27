import numpy as np



def sigmoid(z):
    
    z = np.clip(z, -500, 500)  
    s = 1 / (1 + np.exp(-z))
    return s
    


def initialize_with_zeros_wnb(dim):
   
    w = np.zeros((dim, 1))
    b = 0.0
    
    return w, b


