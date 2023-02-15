from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.backend import tile, repeat, repeat_elements
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf


class AstAttentionGRUModel:
    def __init__(self, config):
        
        config['tdatlen'] = 50
        config['smllen'] = 100
        config['comlen'] = 13
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        
        self.embdims = 100
        self.smldims = 10
        self.recdims = 256

        self.config['num_input'] = 3
        self.config['num_output'] = 1

    def create_model(self):
        #三个输入层，分别对应代码/原文序列、注释序列、扁平化后的AST序列
        dat_input = Input(shape=(self.tdatlen,))
        com_input = Input(shape=(self.comlen,))
        sml_input = Input(shape=(self.smllen,))
        #嵌入层输出的形状是（batch_size, txtvocabsize, embdims）
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)
        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(sml_input)
        #带有256个rnndims单元的GRU层来编码AST，同时使用CuDNNGRU来提升训练速度
        se_enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)

        enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee, initial_state=state_sml)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=state_h)
        #注意力机制，取解码器和code/text编码器对输出作点积
        # 使用注意力向量创建上下文向量
        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)
        context = dot([attn, encout], axes=[2, 1])
        
        ast_attn = dot([decout, seout], axes=[2, 2])
        ast_attn = Activation('softmax')(ast_attn)
        ast_context = dot([ast_attn, seout], axes=[2, 1])

        context = concatenate([context, decout, ast_context])

        out = TimeDistributed(Dense(self.recdims, activation="relu"))(context)
        #扁平化（13, 256）的矩阵，成为（3328）的向量
        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input, sml_input], outputs=out)

        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model
