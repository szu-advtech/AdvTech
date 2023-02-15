import argparse
import torch
import datetime
import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm
import os
from caserpp import CaserPP
from generate_input import get_input
from utility import calculate_hit_ndcg


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
# data arguments
parser.add_argument('--data', nargs='?', default='../datasets/ML1M/data',
                    help='data directory')
# train arguments
parser.add_argument('--epoch', type=int, default=100,
                    help='Number of max epochs.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size.')
parser.add_argument('--emb_size', type=int, default=64,
                    help='Number of hidden factors, i.e., embedding size.')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--nh', type=int, default=16,
                    help='Number of filters per horizontal filter size')
parser.add_argument('--nv', type=int, default=4,
                    help='Number of filters per vertical filter size')
parser.add_argument('--filter_sizes', nargs='?', default='[2, 3, 4]',
                    help='Specify the filter_size')
parser.add_argument("--num_hidden_layers", default=3, type=int, help="number of filter-enhanced blocks")
parser.add_argument("--filter_dropout_prob", default=0.5, type=float)
parser.add_argument("--hidden_act", default="gelu", type=str)  
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument("--initializer_range", default=0.02, type=float)
parser.add_argument('--early_stop_epoch', default=20, type=int)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--l2', type=float, default=1e-6)
parser.add_argument('--is_test', type=str2bool, default=True)
parser.add_argument('--type', type=str, default='all')

args = parser.parse_args()

args.cuda = torch.cuda.is_available()
# use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def predict(args, model, epoch_num, topk, batch_size, _device):
    start_time = datetime.datetime.now()
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    user_num = eval_data.shape[0]
    print("eval user_num: ", user_num)
    with tqdm(total=user_num//batch_size) as t:
        for start in range(0, user_num, batch_size):
            end = start + batch_size if start + batch_size < user_num else user_num
            if start % 5000 == 0:
                print("test progress:{}/{}".format(start, user_num))
            batch = eval_data.iloc[start:end]
            eval_inputs = get_input(args, batch, _device, is_train=False)
            eval_input, target = eval_inputs[:-1], eval_inputs[-1]
            prediction = model(eval_input)
            prediction = prediction.data.cpu().numpy()
            prediction = np.argsort(prediction)

            calculate_hit_ndcg(prediction, topk, target, hit_purchase, ndcg_purchase)
            sleep(0.1)
            t.update(1)

    print('#############################################################')
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


if __name__ == '__main__':

    data_folder = args.data
    data_name = args.data.split('/')[2]

    # statis data
    statis_data = pd.read_pickle(os.path.join(data_folder, 'data_statis.df'))  # includeing seq_len and item_num

    # train data
    train_data = pd.read_pickle(os.path.join(data_folder, 'train.df'))
    # eval data
    is_test = args.is_test
    data_type = args.type

    if not is_test:
        eval_data = pd.read_pickle(os.path.join(data_folder, 'val.df'))
    else:
        if data_type == 'all':
            eval_data = pd.read_pickle(os.path.join(data_folder, 'test.df'))
        elif data_type == 'clicked':
            eval_data = pd.read_pickle(os.path.join(data_folder, 'test_click.df'))
        elif data_type == 'unclicked':
            eval_data = pd.read_pickle(os.path.join(data_folder, 'test_unclick.df'))

    seq_len = statis_data['state_size'][0]  # the length of history to define the state
    item_num = statis_data['item_num'][0]  # total number of item
    user_num = statis_data['user_num'][0]  # total number of user

    print("user_num: ", user_num)
    print("item_num: ", item_num)
    topk = [5, 10, 20]

    batch_size = args.batch_size
    epoch = args.epoch
    emb_size = args.emb_size
    nh = args.nh
    nv = args.nv
    filter_sizes = args.filter_sizes
    num_hidden_layers = args.num_hidden_layers
    filter_dropout_prob = args.filter_dropout_prob
    dropout_rate = args.dropout_rate
    lr = args.lr
    l2 = args.l2
    early_stop_epoch = args.early_stop_epoch
    cuda = args.cuda

    _device = torch.device('cuda' if args.cuda else 'cpu')

    model = CaserPP(args=args, item_num=item_num, user_num=user_num, seq_len=seq_len).to(_device)

    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = '../model/Caser++/{}/emb_{}_dropout_{}_{}'.format(data_name, args.emb_size, args.dropout_rate, now_time)
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    _loss = torch.nn.CrossEntropyLoss()
    _optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    # print(model.parameters)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    print("data number of click :{} , data number of purchase :{}".format(
        train_data[train_data['is_buy'] == 0].shape[0],
        train_data[train_data['is_buy'] == 1].shape[0],
    ))

    num_rows = train_data.shape[0]
    minibatch = int(num_rows / batch_size)
    total_step = 0
    best_hit_10 = -1

    for epoch_num in range(epoch):

        # TODO: Training
        model.train()
        epoch_loss = 0.0
        # print("Epoch: ", epoch_num + 1)
        # print("==========================================================")
        start_time = datetime.datetime.now()

        with tqdm(total=minibatch) as t:
            for j in range(minibatch):
                # Description will be displayed on the left
                t.set_description('Train Epoch %i' % (epoch_num + 1))
                batch = train_data.sample(n=batch_size).to_dict()
                inputs = get_input(args, batch, _device, is_train=True)
                input, target = inputs[:-1], inputs[-1]
                _optimizer.zero_grad()
                output = model(input)
                target = torch.LongTensor(target).to(_device)
                loss = _loss(output, target)
                epoch_loss += loss
                loss.backward()
                _optimizer.step()
                total_step += 1
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                t.set_postfix(loss=loss.item())
                sleep(0.1)
                t.update(1)
                if total_step % 200 == 0:
                    t.write("Done %dth batch, the loss is %f" % (total_step, loss.item()))

        epoch_loss /= minibatch + 1
        over_time_i = datetime.datetime.now()
        total_time_i = (over_time_i - start_time).total_seconds()
        print('epoch loss: ', epoch_loss.item())

        # TODO: Evaluate
        model.eval()
        hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = predict(args, model, epoch_num, topk, batch_size, _device)
        if hit10 > best_hit_10:
            best_hit_10 = hit10
            count = 0
            save_root = os.path.join(save_dir,
                                     'epoch_{}_hit@10_{:.4f}_ndcg@10_{:.4f}'.format(
                                         epoch_num, hit10, ndcg10))
            isExists = os.path.exists(save_root)
            if not isExists:
                os.makedirs(save_root)
            model_name = 'Caser.pt'
            save_root = os.path.join(save_root, model_name)
            torch.save(model.state_dict(), save_root)

        else:
            count += 1
        if count == args.early_stop_epoch:
            break

