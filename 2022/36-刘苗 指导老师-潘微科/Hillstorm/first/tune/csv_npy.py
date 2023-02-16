import pandas as pd
import numpy as np
if __name__ == "__main__":
    train_data=pd.read_csv('../train_Hillstrom.csv')
    np.save("../train_Hillstrom.npy",train_data)
    test_data = pd.read_csv('../test_Hillstrom.csv')
    np.save("../test_Hillstrom.npy", test_data)
    contrastive_pair = pd.read_csv('../contrastive_pair_Hillstrom.csv')
    np.save("../contrastive_pair_Hillstrom.npy", contrastive_pair)
