import torch
import argparse
import logging
import os, sys
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils

parser = argparse.ArgumentParser(description="trainer")
parser.add_argument("--dataset", type=str, help="specify dataset", default="nl")
parser.add_argument("--model", type=str, help="specify model", default="dnn")

# dataset information
# parser.add_argument("--feature", type=int,  default=1000, help="feature number")
# parser.add_argument("--field", type=int,  default=33,  help="field number")
# parser.add_argument("--data_dir", type=str,  default="data", help="data directory")

# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate" , default=1e-4)
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-5)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=20, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, default="save/", help="model save directory")

# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", help="mlp batch normalization")
parser.add_argument("--cross", type=int, help="cross layer", default=3)

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=-1, help="device info")

args = parser.parse_args()

my_seed = 2022
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class Trainer(object):
    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = trainUtils.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = trainUtils.getOptim(self.network, opt["optimizer"],self.lr, self.l2)
        self.logger = trainUtils.get_log(opt['model'])
    
    def train_on_batch(self, label, data):
        self.network.train()
        self.optim.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        logit = self.network(data)
        #print(logit.shape)
        logloss = self.criterion(logit, label)
        regloss = self.network.reg()
        loss = regloss + logloss
        loss.backward()
        self.optim.step()
        return logloss.item()
    
    def eval_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob
    
    def train(self, epochs):
        step = 0
        cur_auc = 0.0
        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            for feature, label, domain in self.dataloader.get_data("train", batch_size = self.bs):
                #print(feature.shape)
                #print(label.shape)
                #print(domain.shape)
                loss = self.train_on_batch(label, feature)
                train_loss += loss
                step += 1
                if step % 100 == 0:
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                          format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            val_auc, val_loss = self.evaluate("val")
            self.logger.info("[Epoch {epoch:d} | Train Loss:{loss:.6f} | Val AUC:{val_auc:.6f}, Val Loss:{val_loss:.6f}".
                        format(epoch=epoch_idx,  loss=train_loss, val_auc=val_auc, val_loss=val_loss ))
            early_stop = False
            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                te_auc, te_loss = self.evaluate("test")
                self.logger.info("Early stop at epoch {epoch:d}|Test AUC: {te_auc:.6f}, Test Loss:{te_loss:.6f}".
                        format(epoch=epoch_idx, te_auc = te_auc, te_loss = te_loss))
                break
        if not early_stop:
            te_auc, te_loss = self.evaluate("test")
            self.logger.info("Final Test AUC:{te_auc:.6f}, Test Loss:{te_loss:.6f}".format(te_auc=te_auc, te_loss=te_loss))

    def evaluate(self, on:str):
        preds, trues = [], []
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred =  self.eval_on_batch(feature)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

def main():
    sys.path.extend(["./modules", "./dataloader", "./utils"])
    if args.dataset.lower() == "fr":
        field_dim = trainUtils.get_stats("data/fr/stats_2")
        data_dir = "data/fr/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)
    elif args.dataset.lower() == "nl":
        field_dim = trainUtils.get_stats("data/nl/stats_2")
        data_dir = "data/nl/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)
    elif args.dataset.lower() == "nlfr":
        field_dim = trainUtils.get_stats("data/nlfr/stats_2")
        data_dir = "data/nlfr/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)
    model_opt={
        "latent_dim":args.dim, "feat_num":feature, "field_num":field,
        "mlp_dropout":args.mlp_dropout, "use_bn": args.mlp_bn, "mlp_dims":args.mlp_dims,
        "cross":args.cross
        }

    opt={"model_opt":model_opt, "dataset":args.dataset, "model":args.model, "lr":args.lr, "l2":args.l2,
         "bsize":args.bsize, "epoch":args.max_epoch, "optimizer":args.optim, "data_dir":data_dir,
         "save_dir":args.save_dir, "cuda":args.cuda
         }
    print(opt)
    trainer = Trainer(opt)
    trainer.train(args.max_epoch)
    
    
if __name__ == "__main__":
    """
    python trainer.py Criteo DeepFM --feature    
    """
    # path = "data/criteo/stats_2"
    # field, feature  = trainUtils.get_stats(path)
    # print(len(field))
    # print(sum(field))
    main()
