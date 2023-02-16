from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf
from keras import metrics

class AttentionGRUModel:
    def __init__(self, config):
        
        config['tdatlen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        
        self.embdims = 100
        self.recdims = 256

        self.config['num_input'] = 2
        self.config['num_output'] = 1

    def create_model(self):

        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)

        enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=state_h)

        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)

        context = dot([attn, encout], axes=[2,1])
        
        
        context = concatenate([context, decout])
        
        
        out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)

        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input], outputs=out)
        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model
