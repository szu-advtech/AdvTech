import base64
import io

import requests
import torch
from flask import Flask, request

from models.Nets import CNN
from utils.options import args_server_parser
from flask_socketio import SocketIO
from models.Fed import getDis
from utils.models import hexToStateDict, stateDictToHex
from models.Fed import getAlpha, getTauI, normalization
import time

args = args_server_parser()


def initNetGlob():
    if args.model == 'cnn':
        net_glob = CNN(num_classes=args.num_classes, num_channels=args.num_channels)
    else:
        exit('Error: unrecognized model')
    return net_glob


app = Flask(__name__)
socketio = SocketIO(app)
net_glob = initNetGlob()

globalState = {
    'n_d': 0,
    's': [
        [],
        [],
        [],
        []
    ],
    't': [],
    'scores': [],
    'uid': {}
}

weights = []


def merge(uid, address, data):
    print("getAlpha参数: ", 1, int(time.time()), data['t0'], 0.003, 1, data['uid'], data['n_d'], data['s'])
    alpha = getAlpha(1, int(time.time()), data['t0'], 0.003, 1, data['uid'], data['n_d'], data['s'])
    print("此次更新率: " + str(alpha))
    if alpha == 0:
        return
    localStateDict = data['local_state_dict']
    globStateDict = data['global_state_dict']

    for k in globStateDict.keys():
        globStateDict[k] = (1 - alpha) * globStateDict[k] + alpha * localStateDict[k]

    stateDictHex = stateDictToHex(globStateDict)
    s = normalization(data['s'])
    print("getTauI参数: ", uid, data['n_d'], s)
    score = getTauI(uid, data['n_d'], s)
    return {
        'model_state_hex': stateDictHex,
        'score': score,
        'address': address,
        'cur_global_state_dict': globStateDict
    }


@app.post('/newLocalModel/<address>')
def newLocalModel(address):
    data = request.json
    localStateDict = hexToStateDict(data['local_model_hex'])
    global net_glob
    globalStateDict = net_glob.state_dict()
    uid = globalState['uid'][address]
    globalState['s'][3][uid] = getDis(globalStateDict, localStateDict)
    # 进行聚合

    res = merge(uid, address, {
        "local_state_dict": localStateDict,
        "global_state_dict": globalStateDict,
        "s": globalState['s'],
        "n_d": globalState['n_d'],
        "uid": uid,
        "t0": globalState['t'][uid]
    })
    requests.post(args.eth_rpc, json={
        "jsonrpc": "2.0",
        "method": "eth_newGlobalModel",
        "params": [
            res['address'],
            res['model_state_hex']
        ],
        "id": 1
    })
    globalState['t'][uid] = time.time()
    globalState['s'][1][uid] = float(res['score'])
    return '1'


@app.post('/newGlobalModel')
def newGlobalModel():
    data = request.json
    globalModelStateDict = hexToStateDict(data['global_model_hex'])
    global net_glob
    net_glob.load_state_dict(globalModelStateDict)
    return '1'


def register(address, dataSize):
    global globalState
    if address in globalState['uid'].keys():
        return False
    globalState['uid'][address] = globalState['n_d']
    globalState['scores'].append(0)
    globalState['t'].append(int(time.time()))
    globalState['n_d'] = globalState['n_d'] + 1
    globalState['s'][0].append(dataSize)
    globalState['s'][1].append(0.5)
    globalState['s'][2].append(1)
    globalState['s'][3].append(0)
    return True


@app.post('/register')
def handleRegister():
    data = request.json
    ret = register(data['address'], int(data['data_size']))
    if ret:
        return '1'
    else:
        return '0'


@app.get("/getTrainInfo")
def getModelType():
    return {
        'model_name': args.model,
        'num_classes': args.num_classes,
        'num_channels': args.num_channels
    }


if __name__ == '__main__':
    # register('0xf64477d1bB82c772F853813F088FfD9C466b3538', 700)
    # k = getAlpha(1, 1670135542, 1670135515.899362, 0.003, 1, 1, 2,  [[20000, 40000], [0.5, 0.5], [1, 1], [2.9873648640861994, 3.351467460227369]])
    socketio.run(app, allow_unsafe_werkzeug=True, port=args.port)
