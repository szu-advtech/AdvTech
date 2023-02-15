import tensorflow as tf
from Utils import safe_lookup
from GNN import SubLayer_E2N, SubLayer_N2E

FLAGS = tf.flags.FLAGS

class Model(object):
    def __init__(self, input_data, embedding_size,
                 grad_alpha = 1. ,silent_print=False,):

        self.embedding_size = embedding_size
        self.silent_print = silent_print
        self.dropout = FLAGS.dropout
        self.training = tf.placeholder_with_default(tf.convert_to_tensor(False), shape=(), name="training")

        self.layer_num = FLAGS.layer
        self.init_input(input_data)
        self.init_embedding_layer()
        self.init_hypergraph_neural_network(grad_alpha)


    def init_input(self, input_data):
        (E4N, N4E), features, feature_types, num_features, labels, num_classes, \
            train_idx, val_idx, test_idx, train_mask, val_mask, test_mask = input_data

        create_tensor = tf.convert_to_tensor

        # 后续可以考虑增加batch，目前是单图
        with tf.name_scope('input'):
            # adj
            if FLAGS.neighbor > 0:  # 保留三个邻接
                E4N = [e4n[:, :FLAGS.neighbor] for e4n in E4N]
                N4E = [n4e[:, :FLAGS.neighbor] for n4e in N4E]

            self.N4E = [ create_tensor(e4n, dtype=tf.int64, name="E4N_{}".format(i)) for i, e4n in enumerate(E4N) ]
            self.E4N = [ create_tensor(n4e, dtype=tf.int64, name="N4E_{}".format(i)) for i, n4e in enumerate(N4E) ]

            # feature
            self.features = [ create_tensor(f, dtype=tf.int64, name=feature_types[i]) for i, f in enumerate(features) ]
            self.mask_feats = [tf.not_equal(f, 0) for f in self.features]
            self.num_node_types = len(features) - 1   # 第一个是超边的
            self.num_nodes = [f.shape[0] for f in features]
            self.feature_types = feature_types
            self.feature_vocab_size = num_features

            # label
            self.labels = create_tensor(labels, dtype=tf.int64, name="labels")
            self.num_classes = num_classes

            # dataset splits
            self.train_idx = create_tensor(train_idx, dtype=tf.int64, name="train_idx")
            self.val_idx = create_tensor(val_idx, dtype=tf.int64, name="val_idx")
            self.test_idx = create_tensor(test_idx, dtype=tf.int64, name="test_idx")
            self.train_mask = tf.cast(create_tensor(train_mask, dtype=tf.bool, name="train_mask"), tf.bool)
            self.val_mask = tf.cast(create_tensor(val_mask, dtype=tf.bool, name="val_mask"), tf.bool)
            self.test_mask = tf.cast(create_tensor(test_mask, dtype=tf.bool, name="test_mask"), tf.bool)

            if not self.silent_print:
                print("Shape of E4N: ", " ".join([str(e.shape) for e in self.E4N]))
                print("Shape of N4E: ", " ".join([str(e.shape) for e in self.N4E]))
                print("Shape of features: ", " ".join([str(f.shape) for f in self.features]))
                print("Type of features: ", self.feature_types)
                print("Num of features: ", self.feature_vocab_size)
                print("Num of classes: ", self.num_classes)


    def init_embedding_layer(self):
        with tf.variable_scope('embedding'):
            self.feature_embedding = tf.get_variable("feature_embedding",     # feature_vocab_size * D
                                                     shape=[self.feature_vocab_size, self.embedding_size],
                                                     initializer=tf.truncated_normal_initializer(),)
            # (N_n, N_f) = > (N_n, N_f, D)
            self.hyperedge_init_embedding = safe_lookup(
                self.feature_embedding, self.features[0], name="e_init_embedding"
            )
            # 注意 第一行是padding，所以得到的embedding每行都相同，是正常的现象。
            self.nodes_init_embedding = [
                safe_lookup(self.feature_embedding, f,
                    name="{}_init_embedding".format(self.feature_types[i+1].split("_")[0]))
                for i, f in enumerate(self.features[1: ])
            ]


    def init_hypergraph_neural_network(self, grad_alpha):    # grad_alpha: scalar
        grad = (FLAGS.coef_grad > 0)
        self.first_attn_inputs = None

        with tf.variable_scope('hgnn'):
            self.hyperedge_vec_list, self.hyperedge_mat_list = [], []
            self.node_vec_list, self.node_mat_list = [], []
            self.sub1_attn_weight_list, self.sub2_attn_weight_list = [], []

            # middle layers
            features = [self.hyperedge_init_embedding] + self.nodes_init_embedding
            mask_feats = self.mask_feats
            for layer in range(1, self.layer_num ):  # 前n-1层
                # TODO: 要想通用化attn值和索引的映射，这里 hyperedge_mat 在 simple 的输出，应加一行0，这样可以保持第一行是padding的设定
                sub1_output = SubLayer_N2E(self.E4N, features, mask_feats,
                                           dropout=self.dropout, training=self.training,
                                           scope="layer{}_sub1_N2E".format(layer),
                                           grad_alpha=grad_alpha if grad else 1.)
                (hyperedge_vec, hyperedge_mat), (attn_inputs, attn_weights) = sub1_output
                if self.first_attn_inputs is None:
                    self.first_attn_inputs = attn_inputs
                grad = False
                # edge feature 更新了
                features = [hyperedge_mat] + features[1: ]
                new_edge_mask_feats = tf.cast(tf.ones(tf.shape(hyperedge_mat)[:-1]), tf.bool) \
                                        if (FLAGS.attn in ["simple", "simiter"] and not FLAGS.simple_keepdim) \
                                        else mask_feats[0]
                mask_feats = [new_edge_mask_feats] + mask_feats[1:]
                self.hyperedge_vec_list.append(hyperedge_vec)
                self.hyperedge_mat_list.append(hyperedge_mat)
                self.sub1_attn_weight_list.append(attn_weights)


                sub2_output = SubLayer_E2N(self.N4E, features, mask_feats,
                                           dropout=self.dropout, training=self.training,
                                           scope="layer{}_sub2_E2N".format(layer))
                (node_vecs, node_mats), attn_weights = sub2_output
                # node feature 更新了
                features = features[:1] + node_mats
                new_node_mask_feats = [
                    (tf.cast(tf.ones(tf.shape(node_mat)[:-1]), tf.bool)
                        if (FLAGS.attn in ["simple", "simiter"] and not FLAGS.simple_keepdim) else mask_feats[i+1])
                    for i, node_mat in enumerate(node_mats)
                ]
                mask_feats = mask_feats[:1] + new_node_mask_feats
                self.node_vec_list.append(node_vecs)
                self.node_mat_list.append(node_mats)
                self.sub2_attn_weight_list.append(attn_weights)

            # last layer
            layer = self.layer_num
            if FLAGS.coef_reconst > 0:
                E4N_list = [tf.concat([self.E4N[t], self.neg_E4N_list[t]], axis=0) for t in range(len(self.E4N))]
                self.neg_E_feature = tf.tile(features[0], [FLAGS.negnum, 1, 1])  # 顺序不变
                features = [tf.concat([features[0], self.neg_E_feature], axis=0)] + features[1: ]
                neg_E_mask_feats = tf.tile(mask_feats[0], [FLAGS.negnum, 1])  # 顺序不变
                cache_mask_feats = [tf.concat([mask_feats[0], neg_E_mask_feats], axis=0)] + mask_feats[1: ]
                sub1_output = SubLayer_N2E(E4N_list, features, cache_mask_feats,
                                           dropout=self.dropout, training=self.training,
                                           scope="layer{}_sub1_N2E".format(layer),
                                           grad_alpha=grad_alpha if grad else 1.)
                grad = False
                (hyperedge_vec_pos_and_neg, hyperedge_mat_pos_and_neg), (attn_inputs_pos_and_neg, attn_weights_pos_and_neg) = sub1_output
                if self.first_attn_inputs is None:
                    self.first_attn_inputs = attn_inputs_pos_and_neg

                self.reconstruct_embd = tf.squeeze(hyperedge_vec_pos_and_neg, axis=1)
                self.reconstruct_label = tf.concat([tf.ones(self.pos_num), tf.zeros(self.pos_num * FLAGS.negnum)], axis=0)

                hyperedge_vec = hyperedge_vec_pos_and_neg[: self.pos_num]
                hyperedge_mat = hyperedge_mat_pos_and_neg[: self.pos_num]
                attn_weights = attn_weights_pos_and_neg[: self.pos_num]

            else:
                sub1_output = SubLayer_N2E(self.E4N, features, mask_feats,
                                           dropout=self.dropout, training=self.training,
                                           scope="layer{}_sub1_N2E".format(layer),
                                           grad_alpha=grad_alpha if grad else 1.)
                grad = False
                (hyperedge_vec, hyperedge_mat), (attn_inputs, attn_weights) = sub1_output
                if self.first_attn_inputs is None:
                    self.first_attn_inputs = attn_inputs

            # edge feature 更新了
            features = [hyperedge_mat] + features[1: ]
            new_edge_mask_feats = tf.cast(tf.ones(tf.shape(hyperedge_mat)[:-1]), tf.bool)
            mask_feats = [new_edge_mask_feats] + mask_feats[1:]
            self.hyperedge_vec_list.append(hyperedge_vec)
            self.hyperedge_mat_list.append(hyperedge_mat)
            self.sub1_attn_weight_list.append(attn_weights)
            sub2_output = SubLayer_E2N(self.N4E, features, mask_feats,
                                       dropout=self.dropout, training=self.training,
                                       scope="layer{}_sub2_E2N".format(layer))
            (node_vecs, node_mats), attn_weights = sub2_output
            # node feature 更新了
            features = features[:1] + node_mats
            new_node_mask_feats = [
                (tf.cast(tf.ones(tf.shape(node_mat)[:-1]), tf.bool))
                for i, node_mat in enumerate(node_mats)
            ]
            mask_feats = mask_feats[:1] + new_node_mask_feats
            self.node_vec_list.append(node_vecs)
            self.node_mat_list.append(node_mats)
            self.sub2_attn_weight_list.append(attn_weights)

            if FLAGS.layer_aggr == 'concat':
                self.behavior_emb_concat = tf.concat(self.hyperedge_vec_list, axis=-1, name="behavior_emb")  # (E, 1 or C, D)
                self.entity_emb_concats = [tf.concat(te, axis=-1, name="entity_{}_emb".format(t))
                                           for t, te in enumerate(zip(*self.node_vec_list))]  # (N_t, 1 or C, D)

            else:
                self.behavior_emb_concat = sum(self.hyperedge_vec_list)  # (E, 1 or C, D)
                self.entity_emb_concats = [sum(te) for t, te in enumerate(zip(*self.node_vec_list))]  # (N_t, 1 or C, D)

            if FLAGS.attn == 'channel':
                pass
            else:  # 去掉中间维
                self.behavior_emb_concat = tf.squeeze(self.behavior_emb_concat, axis=1)  # (E, D)
                self.entity_emb_concats = [tf.squeeze(te, axis=1) for te in self.entity_emb_concats]  # (N_t, D)


    def logits(self, entity=False):
        if entity:
            return self.behavior_emb_concat, self.entity_emb_concats
        else:
            return self.behavior_emb_concat

    def reconstruct_emb_and_label(self):
        return self.reconstruct_embd, self.reconstruct_label

    def attn_weights(self):
        with tf.variable_scope("get_attn_weights"):
            sub1_weights = []
            self.target_layer = 0
            if self.target_layer != 0 and FLAGS.attn.startswith('sim'):
                raise NotImplementedError
            # sub layer 1: E4N (N2E)
            weights = self.inverse_attn_or_gradient(self.sub1_attn_weight_list[self.target_layer])  # 先只计算第一层
            self.ori_attn = weights

            # ↑ 第一列是特征padding，为保持embedding的纯粹，这里不移除
            # linear:
            name_post = ""
            self.linear_layer_for_weight = tf.layers.Dense(self.embedding_size, activation=tf.nn.tanh,
                                                           name="linear_for_weight"+name_post, _reuse=tf.AUTO_REUSE)
            weights = self.linear_layer_for_weight(weights)
            sub1_weights.append(weights)

            return sub1_weights


    def inverse_attn_or_gradient(self, attn_or_gard):
        C = 1 if FLAGS.attn.startswith('sim') else self.num_classes
        E_num = self.num_nodes[0]
        # TODO: 多层的话，可以将attention weight们，先乘在一起，然后送到下面进行计算？   ----好像不能，因为是二跳邻居了
        col_idx = [tf.tile(tf.reshape(self.features[0], [E_num, 1, -1]), multiples=[1, C, 1])]  # (E, C, N_E)
        for t in range(self.num_node_types):
            col_idx_t = safe_lookup(self.features[t + 1], self.E4N[t], "sub1_col_idx_{}".format(t))  # (E, N_t, F_Nt)
            col_idx_t = tf.tile(tf.reshape(col_idx_t, [E_num, 1, -1]), multiples=[1, C, 1])  # (E, C, N_t*F_Nt)
            col_idx.append(col_idx_t)
        col_idx = tf.concat(col_idx, axis=-1)  # (E, C, F_E+\sum_t{N_t*F_Nt})
        row_idx = tf.reshape(tf.cast(tf.range(E_num), tf.int64), [-1, 1, 1])  # (E, 1, 1)
        row_idx = tf.tile(row_idx, multiples=[1, C, tf.shape(col_idx)[-1]])  # (E, C, F_E+\sum_t{N_t*F_Nt})
        chn_idx = tf.reshape(tf.cast(tf.range(C), tf.int64), [1, -1, 1])  # (1, C, 1)
        chn_idx = tf.tile(chn_idx, multiples=[E_num, 1, tf.shape(col_idx)[-1]])  # (E, C, F_E+\sum_t{N_t*F_Nt})
        indices = tf.reshape(tf.stack([row_idx, chn_idx, col_idx], axis=-1), [-1, 3])
        values = tf.reshape(attn_or_gard, [-1])  # (E * C * (F_E + \sum_t{N_t * F_Nt}))

        weights = tf.SparseTensor(indices=indices, values=values, dense_shape=[E_num, C, self.feature_vocab_size])
        weights = tf.sparse.to_dense(weights, validate_indices=False)

        return weights
