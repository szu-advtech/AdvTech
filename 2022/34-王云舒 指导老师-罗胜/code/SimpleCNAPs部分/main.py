import os
import pickle

def read_pickle(work_path):
    data_list = []
    with open(work_path, "rb") as f:
            while True:
                try:
                    data = pickle.load(f)    
                    data_list.append(data)
                except EOFError:
                    break
    return data_list

def main():
    pkl_path = 'gesture_metatest_1.pkl'
    data_list = read_pickle(pkl_path)
    print(data_list)
    
if __name__=="__main__":
    main()