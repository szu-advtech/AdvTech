import torch
import argparse
import logging
import os, sys
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
from modules import dualMask

parser = argparse.ArgumentParser(description="dualMask trainer")
parser.add_argument("--dataset", type=str, help="specify dataset", default="nl")
parser.add_argument("--model", type=str, help="specify model", default="dnn")

# dataset information
# parser.add_argument("--feature", type=int, help="feature number", required=True)
# parser.add_argument("--field", type=int, help="field number", required=True)
# parser.add_argument("--data_dir", type=str, help="data directory", required=True)

# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate" , default=3e-5)
parser.add_argument("--l2", type=float, help="L2 regularization", default=3e-5)
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

# mask information
parser.add_argument("--mask_init", type=float, default=0.5, help="mask initial value" )
parser.add_argument("--final_temp", type=float, default=200, help="final temperature")
parser.add_argument("--search_epoch", type=int, default=20, help="search epochs")
parser.add_argument("--rewind_epoch", type=int, default=1, help="rewind epoch")
parser.add_argument("--reg_lambda1_s", type=float, default=1e-8, help="regularization rate")
parser.add_argument("--reg_lambda1_i", type=float, default=1e-8, help="regularization rate")
parser.add_argument("--reg_lambda1_j", type=float, default=1e-8, help="regularization rate")
parser.add_argument("--reg_lambda2", type=float, default=1e-7, help="regularization rate")
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
        self.epochs = opt["search_epoch"]
        self.rewind_epoch = opt["rewind_epoch"]
        self.reg_lambda1_s = opt["lambda1_s"]
        self.reg_lambda1_i = opt["lambda1_i"]
        self.reg_lambda1_j = opt["lambda1_j"]
        self.reg_lambda2 = opt["lambda2"]
        self.temp_increase = opt["final_temp"] ** (1./ (opt["search_epoch"]-1))
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = dualMask.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optim = dualMask.getOptim(self.network, opt["optimizer"], self.lr, self.l2)
        self.logger = trainUtils.get_log(opt['model'])
    
    def train_on_batch(self, label, data, domain, retrain=False):
        self.network.train()
        self.network.zero_grad()
        data, label, domain = data.to(self.device), label.to(self.device), domain.to(self.device)
        logit1, logit2 = self.network(data)
        logloss1 = self.criterion(logit1, label)
        logloss2 = self.criterion(logit2, label)
        regloss1_s = self.reg_lambda1_s * self.network.reg1_s()
        regloss1_i = self.reg_lambda1_i * torch.mean(self.network.reg1_i() * (1 - domain))
        regloss1_j = self.reg_lambda1_j * torch.mean(self.network.reg1_j() * domain)
        regloss2 = self.reg_lambda2 * self.network.reg2()
        if not retrain:
            loss = regloss1_s + regloss1_i + regloss1_j + regloss2 + torch.mean(logloss1 * (1 - domain) + logloss2 * domain)
        else:
            loss = torch.mean(logloss1 * (1 - domain) + logloss2 * domain)
        loss.backward()
        for optim in self.optim:
            optim.step()
        return loss.item()
    
    def eval_on_batch(self, data, domain):
        self.network.eval()
        with torch.no_grad():
            data, domain = data.to(self.device), domain.to(self.device)
            logit1, logit2 = self.network(data)
            if domain[0] == 0:
                logit = logit1
            else:
                logit = logit2
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob
    
    def search(self):
        self.logger.info("ticket:{t}".format(t=self.network.ticket))
        self.logger.info("-----------------Begin Search-----------------")
        for epoch_idx in range(int(self.epochs)):
            train_loss = .0
            step = 0
            if epoch_idx > 0:
                self.network.temp *= self.temp_increase
            if epoch_idx == self.rewind_epoch:
                self.network.checkpoint()
            for feature, label, domain in self.dataloader.get_data("train", batch_size = self.bs):
                loss = self.train_on_batch(label, feature, domain)
                train_loss += loss
                step += 1
                if step % 10 == 0:
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                          format(epoch=epoch_idx, setp=step, loss=loss))
                    # self.logger.info(self.network.temp)
                    # self.logger.info(self.network.mask_embedding.mask_weight_s)
                    self.logger.info(self.network.compute_remaining_weights())
            train_loss /= step
            self.logger.info("Temp:{temp:.6f}".format(temp=self.network.temp))
            val_auc, val_loss = self.evaluate_val("validation")
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))
            rate_s, rate_i, rate_j = self.network.compute_remaining_weights()
            self.logger.info("Feature s remain:{rate_s:.6f} | Feature i remain:{rate_i:.6f} | Feature j remain:{rate_j:.6f}".format(rate_s=rate_s, rate_i=rate_i, rate_j=rate_j))
        test_auc0, test_loss0 = self.evaluate_test("test", "0")
        test_auc1, test_loss1 = self.evaluate_test("test", "1")
        self.logger.info("Test AUC0: {test_auc:.6f}, Test Loss0: {test_loss:.6f}".format(test_auc=test_auc0, test_loss=test_loss0))
        self.logger.info("Test AUC1: {test_auc:.6f}, Test Loss1: {test_loss:.6f}".format(test_auc=test_auc1, test_loss=test_loss1))

    def evaluate_val(self, on: str):
        preds, trues = [], []
        for feature, label, domain in self.dataloader.get_data(on + "0", batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        for feature, label, domain in self.dataloader.get_data(on + "1", batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def evaluate_test(self, on: str, dom: str):
        preds, trues = [], []
        for feature, label, domain in self.dataloader.get_data(on + dom, batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss
    
    def train(self, epochs):
        self.network.ticket=True
        self.network.rewind_weights()
        cur_auc = 0.0
        early_stop = False
        self.optim = dualMask.getOptim(self.network, "adam", self.lr, self.l2)[:1]
        rate_s, rate_i, rate_j = self.network.compute_remaining_weights()
    
        self.logger.info("-----------------Begin Train-----------------")
        self.logger.info("Ticket:{t}".format(t=self.network.ticket))
        self.logger.info("Feature s remain:{rate_s:.6f} | Feature i remain:{rate_i:.6f} | Feature j remain:{rate_j:.6f}".format(rate_s=rate_s, rate_i=rate_i, rate_j=rate_j))
        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            for feature, label, domain in self.dataloader.get_data("train", batch_size = self.bs):
                loss = self.train_on_batch(label, feature, domain, retrain=True)
                train_loss += loss
                step += 1
                if step % 10 == 0:
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                          format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            val_auc, val_loss = self.evaluate_val("validation")
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))
            
            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                test_auc0, test_loss0 = self.evaluate_test("test", "0")
                test_auc1, test_loss1 = self.evaluate_test("test", "1")
                self.logger.info(
                    "Early stop at epoch {epoch:d} | Test AUC0: {test_auc:.6f}, Test Loss0:{test_loss:.6f}".format(
                        epoch=epoch_idx, test_auc = test_auc0, test_loss = test_loss0))
                self.logger.info(
                    "Early stop at epoch {epoch:d} | Test AUC1: {test_auc:.6f}, Test Loss1:{test_loss:.6f}".format(
                        epoch=epoch_idx, test_auc=test_auc1, test_loss=test_loss1))
                break
        
        if not early_stop:
            test_auc0, test_loss0 = self.evaluate_test("test", "0")
            test_auc1, test_loss1 = self.evaluate_test("test", "1")
            self.logger.info("Final Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}".format(test_auc=test_auc0, test_loss=test_loss0))
            self.logger.info("Final Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}".format(test_auc=test_auc1, test_loss=test_loss1))


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
    elif args.dataset.lower() == "nlfr_v":
        field_dim = trainUtils.get_stats("data/nlfr_v/stats_2")
        data_dir = "data/nlfr_v/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)
    model_opt={
        "latent_dim":args.dim, "feat_num":feature, "field_num":field,
        "mlp_dropout":args.mlp_dropout, "use_bn": args.mlp_bn, "mlp_dims":args.mlp_dims,
        "mask_initial":args.mask_init,"cross":args.cross
        }

    opt={
        "model_opt":model_opt, "dataset":args.dataset, "model":args.model, "lr":args.lr, "l2":args.l2,
        "bsize":args.bsize, "optimizer":args.optim, "data_dir":data_dir,"save_dir":args.save_dir,
        "cuda":args.cuda, "search_epoch":args.search_epoch, "rewind_epoch": args.rewind_epoch,"final_temp":args.final_temp,
        "lambda1_s":args.reg_lambda1_s, "lambda1_i":args.reg_lambda1_i, "lambda1_j":args.reg_lambda1_j, "lambda2":args.reg_lambda2
        }
    print(opt)
    trainer = Trainer(opt)
    trainer.search()
    trainer.train(args.max_epoch)
    
if __name__ == "__main__":
    """
    python trainer.py Criteo DeepFM --feature    
    """
    main()
