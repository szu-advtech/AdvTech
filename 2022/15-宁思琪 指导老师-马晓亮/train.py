import sys
import argparse

from tqdm import tqdm
from loguru import logger

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataloader import TrainDataset, TestDataset, DevDataset, load_sts_data, load_sts_data_unsup, random_swap_word,random_delete_word, load_data_sup, TrainDataset_sup
from model import SimcseModel_dropout, simcse_unsup_loss, SimcseModel_sup, simcse_sup_loss
from transformers import BertModel, BertConfig, BertTokenizer
import matplotlib.pyplot as plt

# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
def show(file1,file2):
    input_1 = torch.load(file1)
    input_2 = torch.load(file2)
    corrcoef_update1 = [0.0]
    corrcoef_update2 = [0.0]

    for line in input_1:
        corrcoef_update1.append(line)
    for line in input_2:
        corrcoef_update2.append(line)

    plt.ion()
    for i in range(1, len(corrcoef_update2)):
        ix = corrcoef_update1[:i]
        iy = corrcoef_update2[:i]
        plt.cla()
        plt.plot(ix)
        plt.plot(iy)
        # plt.title("loss")
        # plt.plot(ix, iy)
        # plt.xlabel("epoch")
        # plt.ylabel("acc")
        plt.pause(0.1)
    plt.ioff()
    plt.show()

    # plt.figure('frame time')
    # plt.subplot(211)
    # plt.plot(corrcoef_update1, '.r', )
    # plt.grid(True)
    # plt.subplot(212)
    # plt.plot(corrcoef_update1)
    # plt.plot(corrcoef_update2)
    # plt.grid(True)
    # plt.show()

def train_unsup(model, train_dl, dev_dl, optimizer, device, save_path1,save_path2):
    """模型训练函数"""
    model.train()
    best = 0
    loss_list = []
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(device)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(device)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(device)

        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_unsup_loss(out, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dl)
            model.train()
            if best < corrcoef:
                best = corrcoef
                torch.save(model.state_dict(), save_path1)
                torch.save(loss_list, save_path2)
                logger.info(f"in batch: {batch_idx} save model,higher corrcoef: {best:.4f} ")
                loss_list.append(best)


def train_sup(model, train_dl, dev_dl, optimizer, save_path, batch_size):
    """模型训练函数
    """
    model.train()
    best = 0
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dl)
            model.train()
            if best < corrcoef:
                early_stop_batch = 0
                best = corrcoef
                torch.save(model.state_dict(), save_path)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                continue
            early_stop_batch += 1
            if early_stop_batch == 10:
                logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                logger.info(f"train use sample number: {(batch_idx - 10) * batch_size}")
                return

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
def eval(model, dataloader) -> float:
    """模型评估函数
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def main(args):
    args.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    train_path_sp = args.data_path + "cnsd-sts-train.txt"
    train_path_unsp = args.data_path + "cnsd-sts-train_unsup.txt"
    dev_path_sp = args.data_path + "cnsd-sts-dev.txt"
    test_path_sp = args.data_path + "cnsd-sts-test.txt"

    dev_data_source = load_sts_data(dev_path_sp)
    test_data_source = load_sts_data(test_path_sp)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    if args.un_supervise:
        train_data_source = load_sts_data_unsup(train_path_unsp)
        sentence = [data[0] for data in train_data_source]
        if args.addreverse:
                train_sents = random_swap_word(sentence, 0.1)
        else:
            if args.addremove:
                train_sents = random_delete_word(sentence, 0.1)
            else:
                train_sents = [data[0] for data in train_data_source]
        train_dataset = TrainDataset(train_sents, tokenizer, max_len=args.max_length)
    else:
        train_data_source = load_data_sup(train_path_sp)
        # train_sents = [data[0] for data in train_data_source] + [data[1] for data in train_data_source]
        train_dataset = TrainDataset_sup(train_data_source)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    test_dataset = TestDataset(test_data_source, tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    dev_dataset = DevDataset(dev_data_source, tokenizer, max_len=args.max_length)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"]

    if args.un_supervise:
        model_drop = SimcseModel_dropout(pretrained_model=args.pretrain_model_path, pooling=args.pooler,
                                         dropout=args.dropout).to(args.device)
        optimizer = torch.optim.AdamW(model_drop.parameters(), lr=args.lr)
        train_unsup(model_drop, train_dataloader, test_dataloader, optimizer, args.device, args.save_path1, args.save_path4)
        model_drop.load_state_dict(torch.load(args.save_path1))
        dev_corrcoef = eval(model_drop, dev_dataloader)
        test_corrcoef = eval(model_drop, test_dataloader)
        logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
        logger.info(f'test_corrcoef: {test_corrcoef:.4f}')
    else:
        model_sup = SimcseModel_sup(pretrained_model=args.pretrain_model_path, pooling=args.pooler).to(args.device)
        optimizer = torch.optim.AdamW(model_sup.parameters(), lr=args.lr)
        train_sup(model_sup, train_dataloader, test_dataloader, optimizer, args.save_path, batch_size=args.batch_size)
        model_sup.load_state_dict(torch.load(args.save_path))
        dev_corrcoef = eval(model_sup, dev_dataloader)
        test_corrcoef = eval(model_sup, test_dataloader)

        logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
        logger.info(f'test_corrcoef: {test_corrcoef:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', help="gpu or cpu")
    parser.add_argument("--save_path1", type=str, default='../model_save/simcse_unsup.pt')
    parser.add_argument("--save_path2", type=str, default='../model_save/train.pth')
    parser.add_argument("--save_path3", type=str, default='../model_save/train_none.pth')
    parser.add_argument("--save_path4", type=str, default='../model_save/train_test.pth')
    parser.add_argument("--un_supervise", type=bool, default=True)
    parser.add_argument("--addreverse", type=bool, default=False)
    parser.add_argument("--addremove", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=float, default=64)
    parser.add_argument("--max_length", type=int, default=64, help="max length of input sentences")
    parser.add_argument("--data_path", type=str, default="../data/STS-B/")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="bert-base-chinese")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='cls', help='which pooler to use')

    args = parser.parse_args()
    # logger.add("../log/train.log")
    # logger.info(args)
    main(args)

    # show(args.save_path2,args.save_path3)
