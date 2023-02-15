from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.backend import tile, repeat, repeat_elements, squeeze, transpose
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

from models.attendgru import AttentionGRUModel
from models.ast_attendgru_xtra import AstAttentionGRUModel as xtra

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attendgru':
        mdl = AttentionGRUModel(config)
    elif modeltype == 'ast-attendgru':
        mdl = xtra(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
