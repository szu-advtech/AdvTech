import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import string

from tqdm import tqdm
from time import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from model import SimcseModel_dropout, simcse_unsup_loss, SimcseModel_sup, simcse_sup_loss
from dataloader import TrainDataset, TestDataset, DevDataset, load_sts_data, load_sts_data_unsup
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    # [batch_size * 2, 1, 768] * [1, batch_size * 2, 768] = [batch_size * 2, batch_size * 2]
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = sim / temp  # 相似度矩阵除以温度系数
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return y_true

def search(model, cons_dataloader, downstream_dataloader, top_k, device):
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    sentence_list = []
    with torch.no_grad():
        for source in tqdm(cons_dataloader):
            x_real_batch_num = source.get('input_ids').shape[0]
            x_input_ids = source.get('input_ids').view(x_real_batch_num * 2, -1).to(device)
            x_attention_mask = source.get('attention_mask').view(x_real_batch_num * 2, -1).to(device)
            x_token_type_ids = source.get('token_type_ids').view(x_real_batch_num * 2, -1).to(device)

            x_out = model(x_input_ids, x_attention_mask, x_token_type_ids)
            x_true = simcse_unsup_loss(x_out, device)
        for source in tqdm(downstream_dataloader):
            y_real_batch_num = source.get('input_ids').shape[0]
            y_input_ids = source.get('input_ids').view(y_real_batch_num * 2, -1).to(device)
            y_attention_mask = source.get('attention_mask').view(y_real_batch_num * 2, -1).to(device)
            y_token_type_ids = source.get('token_type_ids').view(y_real_batch_num * 2, -1).to(device)
            y_out = model(y_input_ids, y_attention_mask, y_token_type_ids)
            y_true = simcse_unsup_loss(y_out, device)
            if top_k != 0:
                if y_true == x_true:
                    sentence_list = source
                    top_k = top_k - 1
    return sentence_list

def run_simcse_demo(port, args):
    app = Flask(__name__, static_folder='./static')
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    CORS(app)

    args.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    sentence_path = args.example_sentences
    query_path = args.example_query

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    downstream_data_source = load_sts_data_unsup(sentence_path)
    downstream_dataset = TrainDataset(downstream_data_source, tokenizer, max_len=args.max_length)
    downstream_dataloader = DataLoader(downstream_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    cons_data_source = load_sts_data_unsup(query_path)
    cons_dataset = TrainDataset(cons_data_source, tokenizer, max_len=args.max_length)
    cons_dataloader = DataLoader(cons_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    model_drop = SimcseModel_dropout(pretrained_model=args.pretrain_model_path, pooling=args.pooler,
                                     dropout=args.dropout).to(args.device)
    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.route('/api', methods=['GET'])
    def api():
        start = time()
        results = search(model_drop, cons_dataloader, downstream_dataloader, top_k=5, device=args.device)
        ret = []
        out = {}
        for sentence, score in results:
            ret.append({"sentence": sentence, "score": score})
        span = time() - start
        out['ret'] = ret
        out['time'] = "{:.4f}".format(span)
        return jsonify(out)

    @app.route('/files/<path:path>')
    def static_files(path):
        return app.send_static_file('files/' + path)
        
    @app.route('/get_examples', methods=['GET'])
    def get_examples():
        with open(query_path, 'r') as fp:
            examples = [line.strip() for line in fp.readlines()]
        return jsonify(examples)
    
    addr = args.ip + ":" + args.port
    logger.info(f'Starting Index server at {addr}')
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    IOLoop.instance().start()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='cls', help='which pooler to use')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--example_query', default='./static/example_query.txt', type=str)
    parser.add_argument('--example_sentences', default='./static/example_sentence.txt', type=str)
    parser.add_argument('--port', default='', type=str)
    parser.add_argument('--ip', default='http://127.0.0.1')
    parser.add_argument('--load_light', default=False, action='store_true')
    parser.add_argument("--batch_size", type=float, default=64)
    parser.add_argument("--max_length", type=int, default=64, help="max length of input sentences")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="bert-base-chinese")
    parser.add_argument("--dropout", type=float, default=0.15)
    args = parser.parse_args()

    run_simcse_demo(args.port, args)