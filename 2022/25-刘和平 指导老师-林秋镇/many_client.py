from client import FederatedClient
import datasource
from multiprocessing import Pool
from Paras import MIN_NUM_WORKERS_P, VALUE_DATA_CLASS_TRAIN_SIZE_P, GPU_MEMORY_FRACTION_P
import os


def init():
    import tensorflow as tf
    global tf
    global sess
    global keras
    global model_from_json
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"  # （代表仅使用第0，1号GPU）

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # by_cy_gpu
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION_P  # by_cy_gpu
    sess = tf.compat.v1.Session(config=config)


def start_client(value_data_class_train_size):
    print("start client")
    c = FederatedClient(value_data_class_train_size, "127.0.0.1", 5000, datasource.Mnist)


if __name__ == '__main__':

    p = Pool(MIN_NUM_WORKERS_P,initializer=init())
    p.map(start_client, VALUE_DATA_CLASS_TRAIN_SIZE_P)

