import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train, models
from modules.HistoryLog import LossHistory

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='PMR',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
# 当在终端运行的时候，如果不加入--vonly，那么程序running的时候，vonly的值为default: False
# 如果加上了--vonly，不需要指定True/False,那么程序running的时候，vonly的值为True
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                     help='consider aligned experiment or not (default: False)')
# parser.add_argument('--aligned', type=bool, default="False",
#                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='iemocap',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.25,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.3,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=1,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
# branch，根据数据集不同，需要选择设置batch_size, clip, lr, num_epochs, when, num_heads, attn_dropout
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=60,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')
parser.add_argument('--checkpointdir', type=str, help='directory to save/read weights', default='checkpoints')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='PMR',
                     help='name of the trial (default: "PMR")')
parser.add_argument('--logs_dir', type=str, default='logs',
                    help='original root of the log files')
parser.add_argument('--log_dir', type=str, default='PMRresults',
                    help='root of the log files')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
# lonely:'use the crossmodal fusion into l (default: False)'
# vlonely:'use the crossmodal fusion into v (default: False)'
# aonly:'use the crossmodal fusion into a (default: False)'
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

# branch：决定使用数据输出的类别
output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

# branch：决定数据集的损失函数
criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")
#dataset is mosei_senti, aligned is false
# branch：在这里需要决定数据集的名称，是否需要使用对齐
train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')


   
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
#'number of layers in the network (default: 5)'
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
#'when to decay learning rate (default: 20)'
hyp_params.when = args.when
#help='number of chunks per batch (default: 1)'
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
# .get() 没有设置key的值或者没有这个key，则返回后面的那个值当作它的value
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')


hyper_para = dict(model=hyp_params.model, optimizer_type=hyp_params.criterion, dataset=hyp_params.dataset,\
                  whether_aligned="aligned" if hyp_params.aligned else "no aligned", Epoch=hyp_params.num_epochs, batch_size=hyp_params.batch_size,\
                Init_lr=hyp_params.lr, Gradient_clip=hyp_params.clip, Crossmodal_Attention_Heads=hyp_params.num_heads, layers=hyp_params.layers)
loss_history = LossHistory(hyp_params, hyper_para)
# loss_history = LossHistory(hyp_params.log_dir, model=model, dataset=hyp_params.dataset, hyper_para=hyper_para, input_shape=train_data[0])


if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader, loss_history)

