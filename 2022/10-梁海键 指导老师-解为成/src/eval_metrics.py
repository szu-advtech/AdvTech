import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)


def eval_mosei_senti(results, truths, exclude_zero=False, loss_txt=None):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    # 返回Pearson乘积矩相关系数。
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    with open(loss_txt, 'a') as f:
        f.write("\n" + "-" * 50 + "\n")
        f.write("MAE: " + str(mae) + "\n")
        f.write("Correlation Coefficient: " + str(corr) + "\n")
        f.write("mult_acc_7: " + str(mult_a7) + "\n")
        f.write("mult_acc_5: " + str(mult_a5) + "\n")
        f.write("F1 score: " + str(f_score) + "\n")
        f.write("Accuracy: " + str(accuracy_score(binary_truth, binary_preds)) + "\n")

        f.write("-" * 50 + "\n")

# true
def eval_mosi(results, truths, exclude_zero=False, loss_txt=None):
    return eval_mosei_senti(results, truths, exclude_zero, loss_txt)


def eval_iemocap(results, truths, single=-1, loss_txt=None):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()

        with open(loss_txt, 'a') as f:
            for emo_ind in range(4):
                f.write("\n" + f"{emos[emo_ind]}: \n")
                test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
                test_truth_i = test_truth[:,emo_ind]
                f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
                acc = accuracy_score(test_truth_i, test_preds_i)
                f.write("  - F1 Score: " + str(f1) + "\n")
                f.write("  - Accuracy: " + str(acc) + "\n")
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        with open(loss_txt, 'a') as f:
            f.write("\n" + f"{emos[single]}: \n")
            test_preds_i = np.argmax(test_preds, axis=1)
            test_truth_i = test_truth
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            f.write("  - F1 Score: " + str(f1) + "\n")
            f.write("  - Accuracy: " + str(acc) + "\n")



