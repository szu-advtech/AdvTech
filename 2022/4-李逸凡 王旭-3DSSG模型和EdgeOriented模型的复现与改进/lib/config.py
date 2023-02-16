import os
import sys
from easydict import EasyDict
import json

CONF = EasyDict()
# scalar
CONF.SCALAR = EasyDict()
CONF.SCALAR.OBJ_PC_SAMPLE = 1000    # or 1024 in pointnet config
CONF.SCALAR.REL_PC_SAMPLE = 3000

CONF.MODEL = EasyDict()
CONF.MODEL.LR_DECAY_STEP = [30, 40, 50]
CONF.MODEL.LR_DECAY_RATE = 0.1
CONF.MODEL.BN_DECAY_STEP = 20
CONF.MODEL.BN_DECAY_RATE = 0.5

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/data/liyifan/3DSSG_Code/3dssg/"
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE)

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# 3RScan data
CONF.PATH.R3Scan = os.path.join(CONF.PATH.DATA, "../../datasets/3RScan")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

CONF.PATH.D3SSG_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_train.json")))["scans"]
CONF.PATH.D3SSG_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "3DSSG_subset/relationships_validation.json")))["scans"]
