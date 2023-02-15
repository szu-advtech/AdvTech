import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

'''
VAEN(I)
'''

class VAE():
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None, optimizer='Adam', user_num = 36918):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed
        self.optimizer = optimizer
        self.user_num = user_num

    def create_params(self):
        # params of q network
        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance, respectively
                d_out *= 2 
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)
            # W
            self.weights_q.append(tf.get_variable(name=weight_key, shape=[
                                  d_in + self.user_num, d_out], initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)))
            # b
            self.biases_q.append(tf.get_variable(name=bias_key, shape=[
                                 d_out], initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))

        # params of p network
        self.weights_p, self.biases_p = [], []
        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.get_variable(name=weight_key, shape=[
                                  d_in, d_out], initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)))

            self.biases_p.append(tf.get_variable(name=bias_key, shape=[
                                 d_out], initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))

    def create_placeholders(self):
        self.input_ph_auxiliary = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.input_ph_target = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.input_ph_union = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.input_neighbor = tf.placeholder(dtype=tf.float32, shape=[None, self.user_num])

        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

        self.prediction_top_k = tf.placeholder(tf.float32, [None, None])
        self.scale_top_k = tf.placeholder(tf.int32)
        self.top_k = tf.nn.top_k(self.prediction_top_k, self.scale_top_k)

    def q_graph(self):
        mu_q, std_q, KL = None, None, None
        
        concat_input = tf.concat((self.input_ph_target, self.input_neighbor), 1)
        h = tf.nn.l2_normalize(concat_input, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b 
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]
                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum(
                    0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z
        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def inference(self):
        # q-network  
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))
        sampled_z = mu_q + self.is_training_ph * epsilon * std_q

        # p-network
        self.logits_target = self.p_graph(sampled_z)
        self.KL = KL

    def create_loss(self):
        log_softmax_var = tf.nn.log_softmax(self.logits_target)
        self.neg_ll = - tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.input_ph_target, axis=-1))

        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)

        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = self.neg_ll + self.anneal_ph * self.KL + 2 * reg_var
        self.loss = neg_ELBO

    def create_optimizer(self):
        if self.optimizer == 'GD':
            self.train_op = tf.train.GradientDescentOptimizer(
                self.lr).minimize(loss=self.loss, var_list=self.params)
        elif self.optimizer == 'Adam':
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer == 'Adagrad':
            self.train_op = tf.train.AdagradOptimizer(
                self.lr).minimize(loss=self.loss, var_list=self.params)

    def build_graph(self):
        self.create_params()
        self.create_placeholders()
        self.inference()
        self.create_loss()
        self.create_optimizer()
