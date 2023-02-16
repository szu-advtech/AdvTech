from absl import flags
import sys, os
import time, random
import collections
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from utils.train_help import get_supernet, get_log, get_cuda, get_optimizer, get_stats, get_dataloader

my_seed = 0
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)
random.seed(my_seed)

FLAGS = flags.FLAGS
flags.DEFINE_integer("gpu", 0, "specify gpu core", lower_bound=-1, upper_bound=7)
flags.DEFINE_string("dataset", "KDD12", "Criteo, Avazu or KDD12")

# Mode
flags.DEFINE_string("mode_supernet", "none", "mode to use: feature for optembedding-f, embed for optembedding-e, all for optembedding, none for neither")
flags.DEFINE_string("mode_threshold", "field", "mode to use for assign threshold: feature-level or field-level")
flags.DEFINE_string("mode_oov", "zero", "mode for pruned feature: oov or zero")

# General Model
flags.DEFINE_string("model", "deepfm", "prediction model")
flags.DEFINE_integer("log_interval", 1000, "logging interval")
flags.DEFINE_integer("batch_size", 2048, "batch size")
flags.DEFINE_integer("epoch", 30, "epoch for training/pruning")
flags.DEFINE_integer("latent_dim", 64, "latent dimension for embedding table")
flags.DEFINE_list("mlp_dims", [1024, 512, 256], "dimension for each MLP")
flags.DEFINE_float("mlp_dropout", 0.0, "dropout for MLP")
flags.DEFINE_string("optimizer", "adam", "optimizer for training")
flags.DEFINE_float("lr", 3e-5, "model learning rate")
flags.DEFINE_float("wd", 1e-5, "model weight decay")
flags.DEFINE_integer("cross_layer_num", 6, "cross layer num") # Deep & Cross Network

# Learnable threshold
flags.DEFINE_float("t_lr", 1e-4, "threshold learning rate")
flags.DEFINE_float("alpha", 1e-5, "threshold regularization term")
flags.DEFINE_integer("norm", 1, "norm used")

# How to save model
flags.DEFINE_integer("debug_mode", 0, "0 for debug mode, 1 for noraml mode")
flags.DEFINE_string("save_path", "save/", "Path to save")
flags.DEFINE_string("save_name", "model_weight.pth", "Save file name")
flags.DEFINE_string("init_name", "model_init.pth", "Init file name")
FLAGS(sys.argv)

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class SuperNet(object):
    def __init__(self, opt):
        self.mode_supernet = opt['mode_supernet']
        self.mode_threshold = opt['mode_threshold']
        self.mode_oov = opt['mode_oov']
        self.loader = get_dataloader(opt["dataset"], opt["data_path"])
        self.save_path = os.path.join(opt["save_path"], opt['dataset'], opt['model'], opt['mode_supernet'], opt["mode_threshold"], opt["mode_oov"])
        self.save_name = opt["save_name"]
        self.init_name = opt["init_name"]
        self.debug_mode = opt["debug_mode"]
        self.batch_size = opt["batch_size"]
        self.log_interval = opt["log_interval"]
        self.alpha = opt['train']['alpha']
        self.norm = opt['train']['norm']

        if opt['cuda'] != -1:
            get_cuda(True, 0)
            self.device = torch.device('cuda')
            opt['train']['use_cuda']=True
        else:
            self.device = torch.device('cpu')
            opt['train']['use_cuda'] = False
        self.model = get_supernet(opt['train']).to(self.device)
        
        self.criterion = F.binary_cross_entropy_with_logits
        self.optimizer = get_optimizer(self.model, opt["train"])
        self.logger = get_log(self.mode_supernet)

    def __update(self, label, data):
        self.model.train()
        for opt in self.optimizer:
            opt.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        prob = self.model.forward(data, phase='train')
        logloss = self.criterion(prob, label.squeeze())
        if self.mode_supernet in ['all', 'feature']:
            regloss = self.alpha * torch.sum(torch.exp(0-self.model.threshold))
            loss = logloss + regloss
        else:
            loss = logloss
        loss.backward()
        for opt in self.optimizer:
            opt.step()
        return logloss.item()

    def __evaluate(self, label, data):
        self.model.eval()
        data, label = data.to(self.device), label.to(self.device)
        prob = self.model.forward(data, phase='test')
        prob = torch.sigmoid(prob).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        return prob, label

    def eval_one_part(self, name):
        preds, trues = [], []
        for feature,label in self.loader.get_data(name, batch_size=self.batch_size):
            pred, label = self.__evaluate(label, feature)
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = roc_auc_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred)
        return auc, loss

    def __save_model(self, save_name):
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, save_name))

    def train_epoch(self, max_epoch):
        print('-' * 80)
        print('Begin Training ...')
        step_idx = 0
        acc_auc = 0.0
        if self.debug_mode == 1:
            self.__save_model(self.init_name)
        for epoch_idx in range(int(max_epoch)):
            epoch_step = 0
            train_loss = 0.0
            for feature, label in self.loader.get_data("train", batch_size = self.batch_size):
                step_idx += 1
                epoch_step += 1
                update_loss = self.__update(label, feature)
                train_loss += update_loss
                if epoch_step % self.log_interval == 0:
                    if self.mode_supernet in ['all', 'feature', 'all2']:
                        sparsity, params = self.model.calc_sparsity()
                        self.logger.info("[Epoch {epoch:d} | Step {step:d} | Update loss {loss:.6f} | Sparsity {sparsity:.6f} ]".
                                format(epoch=epoch_idx, step=epoch_step, loss=update_loss, sparsity=sparsity))
                        k = self.model.threshold.cpu().detach().numpy()
                        self.logger.info("Threshold: [Max {max:.6f} | Min {min:.6f} | Mean {mean:.6f}]".format(max=np.max(k), min=np.min(k), mean=np.mean(k)))
                        norm = torch.norm(self.model.embedding, self.norm, dim=1).cpu().detach().numpy()
                        self.logger.info("Norm: [Max {max:.6f} | Min {min:.6f} | Mean {mean:.6f}]".format(max=np.max(norm), min=np.min(norm), mean=np.mean(norm)))
                    else:
                        self.logger.info("[Epoch {epoch:d} | Step {step:d} | Update loss {loss:f}]".
                                format(epoch=epoch_idx, step=epoch_step, loss=update_loss))
            train_loss /= epoch_step
            val_auc, val_loss = self.eval_one_part(name='val')
            test_auc, test_loss = self.eval_one_part(name='test')
            if self.mode_supernet in ['all', 'feature']:
                sparsity, params = self.model.calc_sparsity()
                self.logger.info("[Epoch {epoch:d} | Train Loss:{loss:.6f} | Sparsity:{sparsity:.6f} | Params:{params:d}]".
                        format(epoch=epoch_idx, loss=train_loss, sparsity=sparsity, params=params))   
            else:
                sparsity = 0.0
                self.logger.info("[Epoch {epoch:d} | Train Loss: {loss:.6f}]".
                        format(epoch=epoch_idx, loss=train_loss))
            self.logger.info("[Epoch {epoch:d} | Val Loss:{loss:.6f} | Val AUC: {auc:.6f}]".
                    format(epoch=epoch_idx, loss=val_loss, auc=val_auc))
            self.logger.info("[Epoch {epoch:d} | Test Loss:{loss:.6f} | Test AUC: {auc:.6f}]".
                    format(epoch=epoch_idx, loss=test_loss, auc=test_auc))
            
            if acc_auc < val_auc:
                acc_auc, acc_sparsity = val_auc, sparsity
                acc_test_auc, acc_test_logloss = test_auc, test_loss
                if self.debug_mode == 1:
                    self.__save_model(self.save_name)
            else:
                self.logger.info("Early stopped!!!")
                break

        self.logger.info("Most Accurate | AUC: {} | Logloss: {} | Sparsity: {}".format(acc_test_auc, acc_test_logloss, acc_sparsity))

def main():
    sys.path.extend(["./models","./dataloader","./utils"])
    if FLAGS.dataset == "Criteo":
        field_dim = get_stats("data/criteo/stats_2")
        data = "data/criteo/threshold_2"
    elif FLAGS.dataset == "Avazu":
        field_dim = get_stats("data/avazu/stats_2")
        data = "data/avazu/threshold_2"
    elif FLAGS.dataset == "KDD12":
        field_dim = get_stats("data/kdd2012_track2/stats")
        data = "data/kdd2012_track2/tfrecord"
    
    train_opt = {
        "mode_supernet":FLAGS.mode_supernet, "mode_threshold":FLAGS.mode_threshold, "mode_oov":FLAGS.mode_oov,
        "model":FLAGS.model, "optimizer":FLAGS.optimizer, "lr":FLAGS.lr, "wd":FLAGS.wd, 
        "field_dim":field_dim, "latent_dim":FLAGS.latent_dim, 
        "mlp_dims":FLAGS.mlp_dims, "mlp_dropout":FLAGS.mlp_dropout,
        "cross_layer_num":FLAGS.cross_layer_num, 
        "threshold_lr":FLAGS.t_lr, "alpha":FLAGS.alpha, "norm":FLAGS.norm
    }
    opt = {
        "mode_supernet":FLAGS.mode_supernet, "mode_threshold":FLAGS.mode_threshold, "mode_oov":FLAGS.mode_oov,
        "dataset":FLAGS.dataset, "cuda":FLAGS.gpu, "data_path":data, 
        "model":FLAGS.model, "batch_size":FLAGS.batch_size, 
        "log_interval":FLAGS.log_interval, "debug_mode":FLAGS.debug_mode, 
        "save_path":FLAGS.save_path, "save_name":FLAGS.save_name, "init_name":FLAGS.init_name,
        "train":train_opt
    }
    print("opt:{}".format(opt))

    sner = SuperNet(opt)
    sner.train_epoch(FLAGS.epoch)

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
