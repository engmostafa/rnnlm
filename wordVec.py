import numpy as np

class WordVec():

    def __init__(NumToWordDict, d = 100):
        
        self.d = d
        self.shape = (len(NumToWordDict), d)
        self.vectors = np.random.uniform(-1.0,1.0,self.shape)
        self.numToWord = NumToWordDict


