import base64
import io

import torch


def stateDictToBase64(stateDict):
    buffer = io.BytesIO()
    torch.save(stateDict, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read())


def base64ToStateDict(b64):
    bytesData = base64.b64decode(b64)
    buffer = io.BytesIO(bytesData)
    return torch.load(buffer)


def stateDictToHex(stateDict):
    buffer = io.BytesIO()
    torch.save(stateDict, buffer)
    buffer.seek(0)
    return bytes.hex(buffer.read())


def hexToStateDict(hex):
    bytesData = bytes.fromhex(hex)
    buffer = io.BytesIO(bytesData)
    return torch.load(buffer)
