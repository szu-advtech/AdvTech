from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np


def calculate_performace(num, y_pred, y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(num):
        if y_test[index] ==1:
            if y_test[index] == y_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == y_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn)/num
    try:
        precision = float(tp)/(tp + fp)
        recall = float(tp)/ (tp + fn)
        f1_score = float((2*precision*recall)/(precision+recall))
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        sens = tp/(tp+fn)
    except ZeroDivisionError:
        print("You can't divide by 0.")
        precision=f1_score =sens = MCC=100
    AUC = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test,y_prob)

    return round(acc,4), round(precision,4), round(sens,4), round(f1_score,4), round(MCC,4), round(AUC,4), round(auprc,4)