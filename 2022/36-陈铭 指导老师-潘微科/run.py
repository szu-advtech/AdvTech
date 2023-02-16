# Thanks to MaJining92 for some codes referred from
# https://github.com/MaJining92/EMCDR

import pickle
import argparse
from LatentFactorModeling.MF import *
from LatentSpaceMapping.LM import *
from LatentSpaceMapping.MLP import *

random.seed(1209)
np.random.seed(1209)

def setParameters():
    '''
    Set the parameters which will be used for running EMCDR model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd', default='../data/version2/ml_25m_rating.csv',
                        help='data path in the source domain')
    parser.add_argument('--td', default='../data/version2/netflix_rating.csv',
                        help='data path in the target domain')
    parser.add_argument('--LFM', default='BPR', choices=['MF', 'BPR'],
                        help='Latent Factor Model')
    parser.add_argument('--LSM', default='LM', choices=['LM', 'MLP'],
                        help='Latent Space Mapping')
    parser.add_argument('--K', default=100, help='dimension of the latent factor',
                        type=int)
    parser.add_argument('--CSF', default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='cold-start entities fractions', type=list)

    args = parser.parse_args()
    return args

def evaluate():
    pass

if __name__ == '__main__':
    args = setParameters()
    Ms, Mt = load_matrix(args.sd, args.td)

    # args:matrix, k, lr, lamda_U, lamda_V, epochs, domain
    # MF(Ms, args.K, 0.001, 0.001, 0.001, 200)
    # MF(Mt, args.K, 0.001, 0.001, 0.001, 200, 'target')

    source_model_path = './LatentFactorModeling'
    source_meta_path = './LatentFactorModeling'

    target_model_path = './LatentFactorModeling'
    target_meta_path = './LatentFactorModeling'

    if args.LFM == 'MF':
        source_model_path += '/MF/mf_s'
        source_meta_path += '/MF/mf_s/mf_s.ckpt.meta'

        target_model_path += '/MF/mf_t'
        target_meta_path += '/MF/mf_t/mf_t.ckpt.meta'
    elif args.LFM == 'BPR':
        source_model_path += '/BPR/bpr_s'
        source_meta_path += '/BPR/bpr_s/bpr_s.ckpt.meta'

        target_model_path += '/BPR/bpr_t'
        target_meta_path += '/BPR/bpr_t/bpr_t.ckpt.meta'

    # Us∈(100, 98907), Vs∈(100, 4100)
    Us, Vs = load_embedding(source_meta_path, source_model_path)
    # Ut∈(100, 99683), Vt∈(100, 4100)
    Ut, Vt = load_embedding(target_meta_path, target_model_path)

    if args.LSM == 'LM':
        if args.LFM == 'MF':
            # args: input_Vs, input_Vt, alpha, lr, Ut,
            #                   latent_factor_model = 'MF',
            #                   cold_start_entities_fraction=0.2,
            #                   max_epoch = 10000,
            #                   patience_count = 5,
            #                   verbose=True, display_step=100, **kwargs
            LinearMapping(Vs, Vt, 0.001, 0.05, Ut, 'MF', 0.3, rating_matrix=Mt)
        elif args.LFM == 'BPR':
            # load the data dict for BPR model
            with open('./netflix_dict.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            LinearMapping(Vs, Vt, 0.1, 0.0001, Ut, 'BPR', 0.3,
                          patience_count=100, display_step=1, data_dict=data_dict)

    elif args.LSM == 'MLP':
        if args.LFM == 'MF':
            # args:input_Vs, input_Vt, Ut, alpha, lr,
            #      latent_factor_model = 'MF',
            #      cold_start_entities_fraction = 0.2,
            #      max_epoch = 10000,
            #      patience_count = 5,
            #      verbose = True, display_step = 100, ** kwargs
            MultiLayerPerceptron(Vs, Vt, Ut, 0.005, 0.005, 'MF', 0.5, rating_matrix=Mt)
        elif args.LFM == 'BPR':
            # load the data dict for BPR model
            with open('./netflix_dict.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            MultiLayerPerceptron(Vs, Vt, Ut, 0.1, 0.0005, 'BPR', 0.5,
                                 patience_count=150, display_step=1, data_dict=data_dict)



