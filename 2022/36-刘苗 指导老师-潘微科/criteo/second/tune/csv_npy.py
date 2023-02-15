import pandas as pd
import numpy as np
if __name__ == "__main__":
    train_data=pd.read_csv('../train_criteo.csv')
    np.save("../train_criteo.npy", train_data)
    test_data = pd.read_csv('../test_criteo.csv')
    np.save("../test_criteo.npy", test_data)
    contrastive_pair = pd.read_csv('../contrastive_pair_criteo.csv')
    np.save("../contrastive_pair_criteo.npy", contrastive_pair)
