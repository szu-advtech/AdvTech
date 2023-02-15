from absl import flags
import sys, os
import time, random
import collections
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from utils.train_help import get_evo, get_log, get_cuda, get_stats, get_dataloader

my_seed = 0
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)
random.seed(my_seed)

FLAGS = flags.FLAGS
flags.DEFINE_integer("gpu", 0, "specify gpu core", lower_bound=-1, upper_bound=7)
flags.DEFINE_string("dataset", "KDD12", "Criteo, Avazu or KDD12")

# Mode
flags.DEFINE_string("mode_supernet", "all", "mode to use for training supernet: embed for optembedding-e, all for optembedding")
flags.DEFINE_string("mode_threshold", "field", "mode to use for assign threshold: feature-level or field-level")
flags.DEFINE_string("mode_oov", "zero", "mode for pruned feature: oov or zero")

# General Model
flags.DEFINE_string("model", "ipnn", "prediction model")
flags.DEFINE_integer("batch_size", 2048, "batch size")
flags.DEFINE_integer("epoch", 30, "epoch for training/pruning")
flags.DEFINE_integer("latent_dim", 64, "latent dimension for embedding table")
flags.DEFINE_list("mlp_dims", [1024, 512, 256], "dimension for each MLP")
flags.DEFINE_float("mlp_dropout", 0.0, "dropout for MLP")

# AutoInt
flags.DEFINE_boolean("has_residual", True, "has residual")
flags.DEFINE_boolean("full_part", True, "full part")
flags.DEFINE_integer("num_heads", 2, "number of headers")
flags.DEFINE_integer("num_layers", 3, "number of layers")
flags.DEFINE_integer("atten_embed_dim", 64, "attention embedding dimension")
flags.DEFINE_float("att_dropout", 0, "attention dropout")

# Deep & Cross Network
flags.DEFINE_integer("cross_layer_num", 6, "cross layer num")

# Evolutionary Search
flags.DEFINE_integer("keep_num", 0, "keep number")
flags.DEFINE_integer("mutation_num", 10, "mutation number")
flags.DEFINE_integer("crossover_num", 10, "crossover_num")
flags.DEFINE_float("m_prob", 0.1, "Mutation Probability")
flags.DEFINE_integer("norm", 1, "norm used")

# How to save model
flags.DEFINE_integer("debug_mode", 1, "0 for debug mode, 1 for noraml mode")
flags.DEFINE_string("supernet_file", "save/Criteo_deepfm_all_field/best_supernet.pth", "Supernet file")
flags.DEFINE_string("save_path", "save/", "Path to save")
flags.DEFINE_string("save_name", "best_arch.pth", "Save file name")
flags.DEFINE_string("supernet_name", "model_weight.pth", "Supernet file")
FLAGS(sys.argv)

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class EvolutionSearcher(object):
    def __init__(self, opt):
        self.mode_supernet = opt['mode_supernet']
        self.mode_threshold = opt['mode_threshold']
        self.mode_oov = opt['mode_oov']
        self.loader = get_dataloader(opt["dataset"], opt["data_path"])
        self.save_path = os.path.join(opt["save_path"], opt['dataset'], opt['model'], opt['mode_supernet'], opt["mode_threshold"], opt["mode_oov"])
        self.save_name = opt["save_name"]
        self.debug_mode = opt["debug_mode"]
        self.batch_size = opt["batch_size"]
        self.latent_dim = opt['train']['latent_dim']
        self.field_num = len(opt['train']['field_dim'])

        if opt['cuda'] != -1:
            get_cuda(True, 0)
            self.device = torch.device('cuda')
            opt['train']['use_cuda']=True
        else:
            self.device = torch.device('cpu')
            opt['train']['use_cuda'] = False
        self.model = get_evo(opt['train']).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, opt['supernet_name'])), strict=False)
        self.model.prepare_sparse_feature()

        # Evolutionary Search Hyper-params
        self.population_num = opt['keep_num'] + opt['mutation_num'] + opt['crossover_num']
        self.keep_num = opt['keep_num']
        self.mutation_num = opt['mutation_num']
        self.crossover_num = opt['crossover_num']
        self.m_prob = opt['m_prob']
        
        self.logger = get_log("evolution")

    def calc_all_params(self):
        params = []
        for cand in self.cands:
            _, param = self.model.calc_sparsity(cand)
            params.append(param)
        return params

    def __save_model(self, cand):
        os.makedirs(self.save_path, exist_ok=True)
        feature_mask = self.model.calc_feature_mask()
        embed_mask = cand
        save_dict = collections.OrderedDict([("feature_mask", feature_mask), ("embed_mask", embed_mask)])
        torch.save(save_dict, os.path.join(self.save_path, self.save_name))

    def __evaluate(self, label, data, cand):
        self.model.eval()
        data, label = data.to(self.device), label.to(self.device)
        prob = self.model.forward(data, cand)
        prob = torch.sigmoid(prob).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        return prob, label

    def eval_one_part(self, name, cand):
        preds, trues = [], []
        for inputs, label in self.loader.get_data(name, batch_size=self.batch_size):
            pred, label = self.__evaluate(label, inputs, cand)
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = roc_auc_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred)
        return auc, loss

    def eval_all_parts(self, name):
        aucs, losses = [], []
        for i, cand in enumerate(self.cands):
            auc, loss = self.eval_one_part(name, cand)
            aucs.append(auc)
            losses.append(loss)
        return aucs, losses

    def get_random(self, num):
        print("Generating random embedding masks ...")
        self.cands = []
        for i in range(num):
            cand = torch.randint(low=0, high=self.latent_dim, size=(self.field_num,)).to(self.device)
            self.cands.append(cand)

    def sort_cands(self, metrics):
        reverse = [1-i for i in metrics]
        indexlist = np.argsort(reverse)
        self.cands = [self.cands[i] for i in indexlist]

    def get_mutation(self, mutation_num, m_prob):
        mutation = []
        assert m_prob > 0

        for i in range(mutation_num):
            origin = self.cands[i]
            for i in range(self.field_num):
                if random.random() < m_prob:
                    index = torch.tensor(i).to(self.device)
                    rand_value = torch.randint(low=1, high=self.latent_dim, size=(1,)).to(self.device)
                    origin[index] = rand_value
            mutation.append(origin)
        return mutation

    def get_crossover(self, crossover_num):
        crossover = []

        def indexes_gen(m, n):
            seen = set()
            x, y = random.randint(m, n), random.randint(m, n)
            while True:
                seen.add((x,y))
                yield (x, y)
                x, y = random.randint(m, n), random.randint(m, n)
                while (x, y) in seen:
                    x, y = random.randint(m, n), random.randint(m, n)
        gen = indexes_gen(0, crossover_num)
        
        for i in range(crossover_num):
            point = random.randint(1, self.latent_dim)
            x, y = next(gen)
            origin_x, origin_y = self.cands[x].cpu().numpy(), self.cands[y].cpu().numpy()
            xy = np.concatenate((origin_x[:point], origin_y[point:]))
            crossover.append(torch.from_numpy(xy).to(self.device))   
        return crossover

    def search(self, max_epoch):
        self.logger.info('-' * 80)
        self.logger.info('Begin Searching ...')
        self.get_random(self.population_num)
        acc_auc, acc_param = 0.0, 0.0
        
        for epoch_idx in range(int(max_epoch)):
            aucs, losses = self.eval_all_parts(name='val')
            self.logger.info("Epoch = {} | best AUC {} | worst AUC {}".format(epoch_idx, max(aucs), min(aucs)))
            self.sort_cands(aucs)
            params = self.calc_all_params()
            param_k = np.argmin(params)

            if acc_auc < aucs[0]:
                acc_auc, acc_param, acc_cand = aucs[0], params[0], self.cands[0]

            mutation = self.get_mutation(self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.crossover_num)
            self.cands = self.cands[:self.keep_num] + mutation + crossover

        acc_test_auc, acc_test_logloss = self.eval_one_part(name='test', cand=acc_cand)
        self.logger.info("Most Accurate | AUC: {} | Logloss: {} | Param: {}".format(acc_test_auc, acc_test_logloss, acc_param))
        self.logger.info("Accurate Cand: {}".format(acc_cand))
        if self.debug_mode == 1:
            self.__save_model(acc_cand)
            self.logger.info("Model saved")
    

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
        "model":FLAGS.model, 
        "field_dim":field_dim, "latent_dim":FLAGS.latent_dim,
        "mlp_dims":FLAGS.mlp_dims, "mlp_dropout":FLAGS.mlp_dropout, 
        "has_residual":FLAGS.has_residual, "full_part":FLAGS.full_part, 
        "num_heads":FLAGS.num_heads, "num_layers":FLAGS.num_layers, 
        "atten_embed_dim":FLAGS.atten_embed_dim, "att_dropout":FLAGS.att_dropout, 
        "cross_layer_num":FLAGS.cross_layer_num, "norm":FLAGS.norm
    }
    opt = {
        "mode_supernet":FLAGS.mode_supernet, "mode_threshold":FLAGS.mode_threshold, "mode_oov":FLAGS.mode_oov,
        "dataset":FLAGS.dataset, "cuda":FLAGS.gpu, "data_path":data,
        "model":FLAGS.model, "batch_size":FLAGS.batch_size, 
        "keep_num":FLAGS.keep_num, "mutation_num":FLAGS.mutation_num, 
        "crossover_num":FLAGS.crossover_num, "m_prob":FLAGS.m_prob, 
        "debug_mode":FLAGS.debug_mode, "supernet_name":FLAGS.supernet_name, 
        "save_path":FLAGS.save_path, "save_name":FLAGS.save_name,
        "train":train_opt
    }
    print("opt:{}".format(opt))

    searcher = EvolutionSearcher(opt)
    searcher.search(FLAGS.epoch)

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)