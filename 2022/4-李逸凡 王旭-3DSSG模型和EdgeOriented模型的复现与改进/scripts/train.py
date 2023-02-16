from ast import arg
import os
import sys
import json
import argparse
from threading import local
from xmlrpc.client import Fault
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from graphviz import Digraph
from tqdm import tqdm

os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import SyncBatchNorm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchsummary import summary

sys.path.append(os.path.join(os.getcwd(),"lib")) # HACK add the root folder
from config import CONF
from data.dataset import D3SemanticSceneGraphDataset
from models.sgpn import SGPN
from scripts.solver import Solver
from scripts.eval import get_eval



D3SSG_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_train.json")))["scans"]
D3SSG_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_validation.json")))["scans"]
# D3SSG_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/fix_train.json")))["scans"]
# D3SSG_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/fix_val.json")))["scans"]


# one scan split include at most 9 classes object, used for visualization
node_color_list = ['aliceblue', 'antiquewhite', 'cornsilk3', 'lightpink', 'salmon', 'palegreen', 'khaki',
                   'darkkhaki', 'orange']
WORKERS = 1

def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def read_class(path):
    file = open(os.path.join(CONF.PATH.DATA, path), 'r')
    category = file.readline()[:-1]
    word_dict = []
    while category:
        word_dict.append(category)
        category = file.readline()[:-1]

    return word_dict


class Test_Model(torch.nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Test_Model, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.fc2 = torch.nn.Linear(input_size, output_size)
        self.fc3 = torch.nn.Linear(input_size, output_size)
        self.fc4 = torch.nn.Linear(input_size, output_size)
        self.fc5 = torch.nn.Linear(input_size, output_size)
        self.fc6 = torch.nn.Linear(input_size, output_size)
        self.fc7 = torch.nn.Linear(input_size, output_size)
        self.fc8 = torch.nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())
        return output

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

        print(param)

        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)



def get_model(args, local_rank):
    # initiate model
    use_pretrained_cls = not args.use_pretrained
    torch.cuda.set_device(local_rank)
    model = SGPN(use_pretrained_cls, gconv_dim=128, gconv_hidden_dim=512,
               gconv_pooling='avg', gconv_num_layers=1, mlp_normalization='batch')
    # getModelSize(model)
    # model = Test_Model(1204,1024)


    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained model...")
        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        model.load_state_dict(torch.load(pretrained_path), strict=False)

    # to CUDA
    # model = torch.nn.DataParallel(model)
    # model = model.cuda()
    
    model.cuda(local_rank)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)

    # Use inplace operations whenever possible
    model.apply(inplace_relu)

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader, datasampler, local_rank):
    model = get_model(args, local_rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        if dist.get_rank() == 0:
            os.makedirs(root, exist_ok=True)
            args.root = root

    solver = Solver(
        args = args,
        model=model,
        dataloader=dataloader,
        datasampler=datasampler,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        lr_decay_step=CONF.MODEL.LR_DECAY_STEP,
        lr_decay_rate=CONF.MODEL.LR_DECAY_RATE,
        bn_decay_step=CONF.MODEL.BN_DECAY_STEP,
        bn_decay_rate=CONF.MODEL.BN_DECAY_RATE
    )
    num_params = get_num_params(model)
    print('sgpn params:', num_params)

    return solver, num_params, root

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.all_scan_id)
    info["num_val_scenes"] = len(val_dataset.all_scan_id)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_3dssg(d3ssg_train, d3ssg_val, num_scans, local_rank, origin_rank, num_gpus):
    # get initial scan list
    train_scan_list = sorted(list([data["scan"]+"-"+str(hex(data["split"]))[-1] for data in d3ssg_train]))
    val_scan_list = sorted(list([data["scan"]+"-"+str(hex(data["split"]))[-1] for data in d3ssg_val]))
    if num_scans == -1:
        num_scans = len(train_scan_list)
    else:
        assert len(train_scan_list) >= num_scans

    rank = local_rank - origin_rank
    split = num_scans // num_gpus
    train_scan_list = train_scan_list[rank * split : (rank + 1) * split]

    # slice train_scan_list
    # train_scan_list = train_scan_list[:num_scans]
    ratio = num_scans / len(d3ssg_train)
    val_scan_list = val_scan_list[:int(len(d3ssg_val) * ratio)]

    # filter data in chosen scenes
    new_3dssg_train = []
    for data in d3ssg_train:
        for l in train_scan_list:
            if data["scan"]==l[:-2] and data["split"]==int(l[-1],16):
                new_3dssg_train.append(data)
    new_3dssg_val = []
    for data in d3ssg_val:
        for l in val_scan_list:
            if data["scan"] == l[:-2] and data["split"] == int(l[-1], 16):
                new_3dssg_val.append(data)

    # new_3dssg_val = d3ssg_val

    print("train on {} samples and val on {} samples".format(len(new_3dssg_train), len(new_3dssg_val)))

    return new_3dssg_train, new_3dssg_val, train_scan_list, val_scan_list

def visualize(data_dict, model, obj_dict, pred_dict):
    dot = Digraph(comment='The Scene Graph')
    dot.attr(rankdir='TB')

    with torch.no_grad():
        data_dict = model(data_dict)

    data, pred_relations = get_eval(data_dict)
    triples = data["triples"][0].cpu().numpy()
    object_id = data["objects_id"][0].cpu().numpy()
    object_cat = data["objects_cat"][0].cpu().numpy()
    object_pred = data["objects_predict"][0].cpu().numpy()

    dot.attr(label='predicted')
    # nodes
    obj_pred_cls = np.argmax(object_pred, axis=1)
    dot.attr('node', shape='oval', fontname='Sans')
    for index in range(len(object_cat)):
        id = str(object_id[index])
        dot.attr('node', fillcolor=node_color_list[index], style='filled')
        pred = obj_pred_cls[index]
        gt = object_cat[index]
        note = obj_dict[pred] + '\n(GT:' + obj_dict[gt] + ')'
        dot.node(id, note)
    # edges
    dot.attr('edge', fontname='Sans', color='red', style='filled')
    for relation in pred_relations[0][:20]:
        s, o, p = relation
        if p == 0:
            continue
        dot.edge(str(s), str(o), pred_dict[p])
    dot.attr('edge', fontname='Sans', color='green', style='filled')
    for item in triples:
        s, o, p = item
        if p == 0:
            continue
        dot.edge(str(s), str(o), pred_dict[p])

    # print(dot.source)
    scan = data_dict["scan_id"][0][:-2]
    split = data_dict["scan_id"][0][-1]
    dot.render(filename=os.path.join(CONF.PATH.BASE, 'vis/{}/scene_graph_{}.gv'.format(scan, split)))


def train(args, local_rank):
    # init training dataset
    print("preparing data...")
    
    d3ssg_train, d3ssg_val, train_scene_list, val_scene_list = get_3dssg(D3SSG_TRAIN, D3SSG_VAL, args.scene_num, local_rank, args.local_rank, args.ngpus_per_node)
    d3ssg = {
        "train": d3ssg_train,
        "val": d3ssg_val
    }

    val_dataset = D3SemanticSceneGraphDataset(relationships=d3ssg["val"],
                                                all_scan_id=val_scene_list, split="val", local_rank=local_rank)

    if args.vis:
        dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                                collate_fn=val_dataset.collate_fn, num_workers=WORKERS)
        use_pretrained_cls = not args.use_pretrained
        model = SGPN(use_pretrained_cls, gconv_dim=128, gconv_hidden_dim=512,
                     gconv_pooling='avg', gconv_num_layers=5, mlp_normalization='batch')
        assert args.use_pretrained
        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        model.load_state_dict(torch.load(pretrained_path), strict=False)
        model = model.cuda()
        model.eval()

        obj_class_dict = read_class("3DSSG_subset/classes.txt")
        pred_class_dict = read_class("3DSSG_subset/relationships.txt")

        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if key != "scan_id":
                    data_dict[key] = data_dict[key].cuda()

            visualize(data_dict, model, obj_class_dict, pred_class_dict)

        print("finished rendering.")
        return

    # training seg
    train_dataset = D3SemanticSceneGraphDataset(relationships=d3ssg["train"],
                                                all_scan_id=train_scene_list, split="train", local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=train_dataset.collate_fn, num_workers=WORKERS, pin_memory=True, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=val_dataset.collate_fn, num_workers=WORKERS, pin_memory=True)
    
    synchronize()

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    datasampler = {
        "train":train_sampler
    }

    print("prossess {} initializing...".format(local_rank))
    solver, num_params, root = get_solver(args, dataloader, datasampler, local_rank)
    if local_rank == 0:
        print("start training...\n")
        save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)
    print("finished training.")

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def distributed_worker(local_rank, dist_url, args):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    # args.rank = args.rank * args.ngpus_per_node + args.gpu
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.ngpus_per_node, rank=local_rank) # world_size表示使用多少个gpu
    print('rank', local_rank, ' use multi-gpus...')

    synchronize()

    torch.cuda.set_device(local_rank)
    
    train(args, local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_num", type=int, help="number of scenes", default=-1)
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=100)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=100)    # train iter
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000)   # val iter
    parser.add_argument("--use_pretrained", default=False, type=str, help="Specify the folder name containing the pretrained module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--vis", action="store_true", help="render visualization result")

    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--ngpus_per_node', default=4, type=int)
    # parser.add_argument("--gpu", type=str, help="gpu", default="0")

    # parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    mode = "train"
    if mode == "debug":
        args.verbose = 10
        args.val_step = 10
        args.ngpus_per_node = 4

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    port = _find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"
    mp.spawn(distributed_worker, nprocs=args.ngpus_per_node, args=(dist_url, args))
    # train(args)