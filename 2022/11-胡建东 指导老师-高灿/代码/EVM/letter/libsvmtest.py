from libsvm.svmutil import *






y, x = svm_read_problem('./train.csv')
m = svm_train(y[:200], x[:200], '-c 15')
p_label, p_acc, p_val =svm_predict(y[:200], x[:200], m)