from absl import flags
import sys, os
import time, random
import collections
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from utils.train_help import get_retrain, get_log, get_cuda, get_optimizer, get_stats, get_dataloader

# my_seed = 0
# torch.manual_seed(my_seed)
# torch.cuda.manual_seed_all(my_seed)
# np.random.seed(my_seed)
# random.seed(my_seed)

FLAGS = flags.FLAGS
flags.DEFINE_integer("gpu", 0, "specify gpu core", lower_bound=-1, upper_bound=7)
flags.DEFINE_string("dataset", "KDD12", "Criteo, Avazu or KDD12")

# Mode
flags.DEFINE_string("mode_supernet", "all", "mode to use: feature for optembedding-f, embed for optembedding-e, all for optembedding, none for neither")
flags.DEFINE_string("mode_threshold", "field", "mode to use for assign threshold: feature-level or field-level")
flags.DEFINE_string("mode_retrain", "weight", "mode to use for retrain: feature & weight or weight only")
flags.DEFINE_string("mode_oov", "zero", "mode for pruned feature: oov or zero")

# General Model
flags.DEFINE_string("model", "ipnn", "prediction model")
flags.DEFINE_integer("log_interval", 1000, "logging interval")
flags.DEFINE_integer("batch_size", 2048, "batch size")
flags.DEFINE_integer("epoch", 30, "epoch for training/pruning")
flags.DEFINE_integer("latent_dim", 64, "latent dimension for embedding table")
flags.DEFINE_list("mlp_dims", [1024, 512, 256], "dimension for each MLP")
flags.DEFINE_float("mlp_dropout", 0.0, "dropout for MLP")
flags.DEFINE_string("optimizer", "adam", "optimizer for training")
flags.DEFINE_float("lr", 1e-5, "model learning rate")
flags.DEFINE_float("wd", 1e-6, "model weight decay")

# AutoInt
flags.DEFINE_boolean("has_residual", True, "has residual")
flags.DEFINE_boolean("full_part", True, "full part")
flags.DEFINE_integer("num_heads", 2, "number of headers")
flags.DEFINE_integer("num_layers", 3, "number of layers")
flags.DEFINE_integer("atten_embed_dim", 64, "attention embedding dimension")
flags.DEFINE_float("att_dropout", 0, "attention dropout")

# Deep & Cross Network
flags.DEFINE_integer("cross_layer_num", 6, "cross layer num")

# Learnable threshold
flags.DEFINE_float("t_lr", 1e-4, "threshold learning rate")
flags.DEFINE_float("alpha", 1e-5, "threshold regularization term")
flags.DEFINE_integer("norm", 1, "norm used")

# How to save model
flags.DEFINE_integer("debug_mode", 0, "0 for debug mode, 1 for noraml mode")
flags.DEFINE_string("save_path", "save/", "Path to save")
flags.DEFINE_string("save_name", "retrain.pth", "Save file name")
flags.DEFINE_string("init_name", "model_weight.pth", "Supernet file")
flags.DEFINE_string("arch_name", "best_arch.pth", "Arch file")
FLAGS(sys.argv)

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class Retrainer(object):
    def __init__(self, opt):
        self.loader = get_dataloader(opt["dataset"], opt["data_path"])
        # self.save_path = os.path.join(opt["save_path"], opt['dataset'], opt['model'], opt['mode_supernet'], opt["mode_threshold"], opt['mode_oov'])
        self.save_path = os.path.join(opt["save_path"], opt['dataset'], opt['model'], opt['mode_supernet'], opt["mode_threshold"], 'zero')
        self.save_name = opt["save_name"]
        self.batch_size = opt["batch_size"]
        self.log_interval = opt["log_interval"]
        self.debug_mode = opt["debug_mode"]
        self.arch_name = opt["arch_name"]
        self.init_name = opt['init_name']
        self.alpha = opt['train']['alpha']
        self.norm = opt['train']['norm']

        if opt['cuda'] != -1:
            get_cuda(True, 0)
            self.device = torch.device('cuda')
            opt['train']['use_cuda']=True
        else:
            opt['train']['use_cuda'] = False
        self.model = get_retrain(opt['train']).to(self.device)
        if self.init_name != 'init.pth':
            init = torch.load(os.path.join(self.save_path, self.init_name))
            # print(init)
            self.model.load_state_dict(init, strict=False)
        if self.arch_name != 'arch.pth':
            arch = torch.load(os.path.join(self.save_path, self.arch_name))
            self.model.update_mask(embed_mask=arch['embed_mask'], feature_mask=arch['feature_mask'])

        self.criterion = F.binary_cross_entropy_with_logits
        self.optimizer = get_optimizer(self.model, opt["train"])
        self.logger = get_log()

    def __update(self, label, data):
        self.model.train()
        for opt in self.optimizer:
            opt.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        prob = self.model.forward(data)
        loss = self.criterion(prob, label.squeeze())
        loss.backward()
        for opt in self.optimizer:
            opt.step()
        return loss.item()

    def __evaluate(self, label, data):
        self.model.eval()
        data, label = data.to(self.device), label.to(self.device)
        prob = self.model.forward(data)
        prob = torch.sigmoid(prob).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        return prob, label

    def eval_one_part(self, name='val'):
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

    def __save_model(self):
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.save_name))

    def eval_from_pretrained(self):
        print('-' * 80)
        print('Begin Evaluating ...')
        params, sparsity = self.model.calc_sparsity()
        self.logger.info("[Params {} | Sparsity {}]".format(params, sparsity))
        val_auc, val_loss = self.eval_one_part(name='val')
        test_auc, test_loss = self.eval_one_part(name='test')
        self.logger.info("[Val Loss:{} | Val AUC: {}]".format(val_loss, val_auc))
        self.logger.info("[Test Loss:{} | Test AUC: {}]".format(test_loss, test_auc))

    def train_epoch(self, max_epoch):
        print('-' * 80)
        print('Begin Training ...')
        sparsity, params = self.model.calc_sparsity()
        self.logger.info("Init: [Params {} | Sparsity {}]".format(params, sparsity))
        step_idx = 0
        best_auc = 0.0
        for epoch_idx in range(int(max_epoch)):
            epoch_step = 0
            train_loss = 0.0
            for feature, label in self.loader.get_data("train", batch_size = self.batch_size):
                step_idx += 1
                epoch_step += 1
                update_loss = self.__update(label, feature)
                train_loss += update_loss
                if epoch_step % self.log_interval == 0:
                    sparsity, params = self.model.calc_sparsity()
                    self.logger.info("[Epoch {epoch:d} | Step {step:d} | Update loss {loss:.6f} | Sparsity {sparsity:.6f} ]".
                                format(epoch=epoch_idx, step=epoch_step, loss=update_loss, sparsity=sparsity))
            train_loss /= epoch_step
            val_auc, val_loss = self.eval_one_part(name='val')
            test_auc, test_loss = self.eval_one_part(name='test')
            sparsity, params = self.model.calc_sparsity()
            self.logger.info("[Epoch {epoch:d} | Train Loss:{loss:.6f} | Sparsity:{sparsity:.6f} | Params:{params:d}]".
                        format(epoch=epoch_idx, loss=train_loss, sparsity=sparsity, params=params))
            self.logger.info("[Epoch {epoch:d} | Val Loss:{loss:.6f} | Val AUC: {auc:.6f}]".
                    format(epoch=epoch_idx, loss=val_loss, auc=val_auc))
            self.logger.info("[Epoch {epoch:d} | Test Loss:{loss:.6f} | Test AUC: {auc:.6f}]".
                    format(epoch=epoch_idx, loss=test_loss, auc=test_auc))

            if best_auc < val_auc:
                best_auc = val_auc
                best_test_auc, best_test_loss = test_auc, test_loss
                if self.debug_mode == 1:
                    self.__save_model()
            else:
                self.logger.info("Early stopped!!!")
                break

        self.logger.info("Most Accurate | AUC: {} | Logloss: {}".format(best_test_auc, best_test_loss))


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
        "mode_supernet":FLAGS.mode_supernet, "mode_threshold":FLAGS.mode_threshold, 
        "mode_retrain":FLAGS.mode_retrain, "mode_oov":FLAGS.mode_oov,
        "model":FLAGS.model, "optimizer":FLAGS.optimizer, "lr":FLAGS.lr, "wd":FLAGS.wd, 
        "field_dim":field_dim, "latent_dim":FLAGS.latent_dim, 
        "mlp_dims":FLAGS.mlp_dims, "mlp_dropout":FLAGS.mlp_dropout,
        "has_residual":FLAGS.has_residual, "full_part":FLAGS.full_part, 
        "num_heads":FLAGS.num_heads, "num_layers":FLAGS.num_layers, 
        "atten_embed_dim":FLAGS.atten_embed_dim, "att_dropout":FLAGS.att_dropout, 
        "cross_layer_num":FLAGS.cross_layer_num, 
        "threshold_lr":FLAGS.t_lr, "alpha":FLAGS.alpha, "norm":FLAGS.norm
    }
    opt = {
        "mode_supernet":FLAGS.mode_supernet, "mode_threshold":FLAGS.mode_threshold,
        "mode_retrain":FLAGS.mode_retrain, "mode_oov":FLAGS.mode_oov,
        "dataset":FLAGS.dataset, "cuda":FLAGS.gpu, "data_path":data, 
        "model":FLAGS.model, "batch_size":FLAGS.batch_size, 
        "log_interval":FLAGS.log_interval, "debug_mode":FLAGS.debug_mode,
        "save_path":FLAGS.save_path, "save_name":FLAGS.save_name, 
        "arch_name":FLAGS.arch_name, "init_name":FLAGS.init_name, 
        "train":train_opt
    }
    print("opt:{}".format(opt))

    rter = Retrainer(opt)
    rter.train_epoch(FLAGS.epoch)

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
