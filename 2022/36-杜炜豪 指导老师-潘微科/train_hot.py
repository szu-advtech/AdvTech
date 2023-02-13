import argparse
import random
import logging
import os
from socket import MSG_PEEK
import time
from datetime import datetime
import ast
from tkinter.tix import Tree
import numpy as np
from sklearn import neighbors
import tensorflow as tf
from dataset import loadTargetData, loadAuxiliaryData, loadTestData, loadNeighborData
import evaluation
import math
from model import VAE
from scipy import sparse


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parser_args():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--model', type=str, default='MultiVAE')
    parser.add_argument('--lr_rate', type=float, default=1e-3)
    parser.add_argument('--reg_scale', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--hiddenDim', nargs='+', type=int, default=100)
    # training parameters
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--path', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='Rec15')
    parser.add_argument('--transaction', type=str, default='target_train')
    parser.add_argument('--examination', type=str, default='auxiliary')
    parser.add_argument('--test', type=str, default='target_valid')
    parser.add_argument('--user_num', type=int, default=36917)
    parser.add_argument('--item_num', type=int, default=9621)
    # other parameters
    parser.add_argument('--total_anneal_steps', type=int, default=200000)
    parser.add_argument('--is_train', type=ast.literal_eval, default=True)
    return parser.parse_args()

def train(args, model, sess, targetData_matrix, auxiliaryData_matrix, userList_train, testDict, userList_test, neighborData_matrix,
          topN=[5, 10, 15, 20]):
    stop_count, anneal_cap, update_count = 0, 0.2, 0.0
    best_prec5, best_rec5, best_f15, best_ndcg5, best_1call5, best_iter = 0., 0., 0., 0., 0., 0

    # saver = tf.train.Saver(max_to_keep=1)

    for epoch in range(args.epoch):
        random.shuffle(userList_train)

        # train
        loss = 0.
        for bnum, st_idx in enumerate(range(0, len(userList_train), args.batch_size)):
            end_idx = min(st_idx + args.batch_size, len(userList_train))
            batchIndex = userList_train[st_idx: end_idx]

            if args.total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = anneal_cap

            A = auxiliaryData_matrix[batchIndex]
            if sparse.isspmatrix(A):
                A = A.toarray()
            T = targetData_matrix[batchIndex]
            if sparse.isspmatrix(T):
                T = T.toarray()
            D = A + T
            N = neighborData_matrix[batchIndex]
            if sparse.isspmatrix(N):
                N = N.toarray()

            feed_dict = {model.input_ph_auxiliary: A, model.input_ph_target: T, model.input_ph_union: D, model.input_neighbor: N,
                         model.keep_prob_ph: 0.5, model.anneal_ph: anneal, model.is_training_ph: 1}
            _, e_loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)

            update_count += 1
            loss += e_loss

        # Evaluation
        precision, recall, f1, ndcg, one_call, mrr = [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], \
                                                     [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]

        for bnum, st_idx in enumerate(range(0, len(userList_test), args.batch_size)):
            end_idx = min(st_idx + args.batch_size, len(userList_test))
            batchIndex = userList_test[st_idx: end_idx]

            A = auxiliaryData_matrix[batchIndex]
            if sparse.isspmatrix(A):
                A = A.toarray()
            T = targetData_matrix[batchIndex]
            if sparse.isspmatrix(T):
                T = T.toarray()
            D = A + T
            N = neighborData_matrix[batchIndex]
            if sparse.isspmatrix(N):
                N = N.toarray()

            testDict_batch = []
            for i in batchIndex:
                testDict_batch.append(testDict[i])

            feed_dict = {model.input_ph_auxiliary: A, model.input_ph_target: T, model.input_ph_union: D, model.input_neighbor: N}
            allRatings_batch = sess.run(model.logits_target, feed_dict=feed_dict)

            allRatings_batch[T.nonzero()] = -np.inf

            _, predictedIndices = sess.run(model.top_k, feed_dict={
                model.prediction_top_k: allRatings_batch, model.scale_top_k: topN[-1]})

            precision_batch, recall_batch, f1_batch, ndcg_batch, one_call_batch, mrr_batch = evaluation.computeTopNAccuracy(
                testDict_batch, predictedIndices, topN)

            for index in range(len(topN)):
                precision[index] += precision_batch[index] / len(userList_test)
                recall[index] += recall_batch[index] / len(userList_test)
                f1[index] += f1_batch[index] / len(userList_test)
                ndcg[index] += ndcg_batch[index] / len(userList_test)
                one_call[index] += one_call_batch[index] / len(userList_test)
                mrr[index] += mrr_batch[index] / len(userList_test)

        print('Epoch: %d Loss: %.4f' % (epoch, loss))
        print('Epoch: %d precision@5: %.4f recall@5: %.4f f1@5: %.4f ndcg@5: %.4f one_call@5: %.4f' % (
            epoch, precision[0], recall[0], f1[0], ndcg[0], one_call[0]))

        logger.info('Epoch: %d Loss: %.4f' % (epoch, loss))
        logger.info('Epoch: %d precision@5: %.4f recall@5: %.4f f1@5: %.4f ndcg@5: %.4f one_call@5: %.4f' % (
            epoch, precision[0], recall[0], f1[0], ndcg[0], one_call[0]))
        logger.info('precision@10/recall@10/f1@10/ndcg@10/one_call@10: %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (
            precision[1], recall[1], f1[1], ndcg[1], one_call[1]))
        logger.info('precision@15/recall@15/f1@15/ndcg@15/one_call@15: %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (
            precision[2], recall[2], f1[2], ndcg[2], one_call[2]))
        logger.info('precision@20/recall@20/f1@20/ndcg@20/one_call@20: %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (
            precision[3], recall[3], f1[3], ndcg[3], one_call[3]))

        if best_ndcg5 < ndcg[0]:
            best_prec5, best_rec5, best_f15, best_ndcg5, best_1call5, best_iter = precision[0], recall[0], f1[0], \
                                                                                  ndcg[0], one_call[0], epoch
            stop_count = 0
            # saver.save(sess, './ckpt/vaecf', global_step=epoch)
        else:
            stop_count += 1
            if stop_count >= args.early_stop:
                # np.savetxt('./ori_rating', allRatings)
                break

    print("End. Best Iteration %d: NDCG@5 = %.4f " % (best_iter, best_ndcg5))
    print("Precision@5: %.4f " % best_prec5)
    print("Recall@5: %.4f " % best_rec5)
    print("F1@5: %.4f " % best_f15)
    print("NDCG@5: %.4f " % best_ndcg5)
    print("1-call@5: %.4f " % best_1call5)

    logger.info("End. Best Iteration %d: NDCG@5 = %.4f " % (best_iter, best_ndcg5))
    logger.info("Precision@5: %.4f " % best_prec5)
    logger.info("Recall@5: %.4f " % best_rec5)
    logger.info("F1@5: %.4f " % best_f15)
    logger.info("NDCG@5: %.4f " % best_ndcg5)
    logger.info("1-call@5: %.4f " % best_1call5)


if __name__ == '__main__':
    args = parser_args()

    targetData_matrix, targetDict, usersNum, itemsNum = loadTargetData(args=args)
    auxiliaryData_matrix, auxiliaryDict = loadAuxiliaryData(args=args)
    neighborData_matrix, neighborDict = loadNeighborData(args=args)  ###
    testDict = loadTestData(args=args)
    userList_train = sorted(list(set(targetDict.keys()).union(set(auxiliaryDict.keys()))))
    userList_test = sorted(testDict.keys())
    args.user_num, args.item_num = usersNum, itemsNum

    p_dims = []
    if type(args.hiddenDim) == list:
        p_dims = args.hiddenDim
        p_dims.append(args.item_num)
    else:
        p_dims.append(args.hiddenDim)
        p_dims.append(args.item_num)

    model = VAE(p_dims=p_dims, lam=args.reg_scale, lr=args.lr_rate, random_seed=98765)
    model.build_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    log_dir = './Log/' + args.dataset + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.%f')

    logger = logging.getLogger('Log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        filename=os.path.join(log_dir,
                              "VAEplusplus_%s_%s_%s_batch%d_hidden%s_reg%.4f_lr%.4f-%s.res" % (
                                  args.dataset, args.test, args.examination, args.batch_size, str(p_dims),
                                  args.reg_scale, args.lr_rate, timestamp)), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(args)

    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    logger.info('Start time: %s' % timestamp)

    train(args, model, sess, targetData_matrix, auxiliaryData_matrix, userList_train, testDict, userList_test, neighborData_matrix)

    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    logger.info('End time: %s' % timestamp)
