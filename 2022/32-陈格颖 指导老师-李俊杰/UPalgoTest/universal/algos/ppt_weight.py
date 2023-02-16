from universal.algo import Algo
import pandas as pd
import numpy as np
import os

class PPT(Algo):
    def __init__(self ,datasetName):
        super(PPT, self).__init__()
        self.path = "D:/M/UPalgoTest/pptWeight/" + datasetName + ".csv"


    def weights(self, S):
        w = pd.read_csv(self.path)
        w = np.array(w)

        return w