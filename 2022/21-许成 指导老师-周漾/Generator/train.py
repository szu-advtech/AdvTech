import os
import model
import pprint
pp = pprint.PrettyPrinter()

from config import opts
from datetime import datetime

if __name__ == '__main__':

    if opts.phase == "train":
        current = datetime.now().strftime("%Y%m%d-%H%M")
        opts.log_dir = os.path.join(opts.log_dir,current)
        if not os.path.exists(opts.log_dir):
            os.makedirs(opts.log_dir)
    #opts.log_dir = "log/20210124-0059"
    print('checkpoints:', opts.log_dir)

    opts.max_epoch = 10000
    opts.epochs = 2000
    opts.choice = 'person'
    opts.bs = 24
    model = model.Model(opts)
    model.train()