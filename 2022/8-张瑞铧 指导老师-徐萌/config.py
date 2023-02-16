import argparse


def get_args():
    parser = argparse.ArgumentParser(description='DSen2_CR model')
    parser.add_argument('--dataset_dir', default="E:\\data\\mini data", help="dataset path")
    parser.add_argument('--show_result', default=1, help="show training result for how many iteration")
    parser.add_argument('--save_model', default=50, help="save training model for how many iteration")
    parser.add_argument('--learning_rate', default=7e-5, help="learning rate")
    parser.add_argument('--epoch', default=5, help="numbers of epoches for training")
    parser.add_argument('--batch_size', default=1, help="batch size")
    parser.add_argument('--device', default=True, help="use GPU or CPU")

    parser.add_argument('--load_model_path', default="output\\net_epoch_36_iteration_6000.pth")
    parser.add_argument('--predict_data_dir', default="E:\\data\\mini val data", help="predict dataset path")

    args = parser.parse_args()
    return args
