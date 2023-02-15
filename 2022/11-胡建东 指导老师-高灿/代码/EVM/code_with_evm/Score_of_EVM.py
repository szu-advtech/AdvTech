import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import Code_of_EVM


def perf_measure(y_actual, y_pred):


    import numpy as np
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_actual[i]==99 and y_actual[i]==y_pred[i]:
           TN += 1
        elif y_actual[i]==99 and y_actual[i]!=y_pred[i]:
           FN +=1
        elif y_actual[i]!=99 and y_actual[i]==y_pred[i]:
           TP +=1
        elif y_actual[i]!=99 and y_actual[i]!=y_pred[i]:
           FP += 1
      
    cm=np.zeros((2,2))
    cm[0][0] = TP
    cm[0][1] = FP
    cm[1][0] = FN
    cm[1][1] = TN
    recognition_accuracy = (TP+TN)/(TP+TN+FP+FN)
    np.set_printoptions(suppress=True)
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fmeasure = 2*precision*recall/(precision+recall)
    return(recognition_accuracy,precision,recall,fmeasure,cm)

if __name__ == '__main__':
    accuracy, predictions, yactual = Code_of_EVM.open_set_evm('../letter/train.csv', '../letter/test.csv')
    recognition_accuracy, precision, recall, fmeasure, cm = perf_measure(yactual, predictions)

    print(
        f'Recognition Accuracy: {recognition_accuracy}, F-Measure: {fmeasure}, Precision: {precision}, Recall: {precision}')


    cm = cm.astype(int)
    plt.figure(figsize=(10, 10))
    sn.set(font_scale=1.4)
    mat_names = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
    mat_vals = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    labels = ["{}\n{}".format(v1, v2) for (v1, v2) in zip(mat_names, mat_vals)]
    labels = np.asarray(labels).reshape(2, 2)
    sn.heatmap(cm, cmap='Blues', annot=labels, xticklabels=['Positive', 'Negative'],
               yticklabels=['Positive', 'Negative'], square=True, fmt='')

    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.savefig('../images/EVM_confusion_matrix.png', dpi=300)
