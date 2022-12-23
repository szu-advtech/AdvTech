
"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""
import config
from Code_of_EVM import load_data, timer, fit, reduce_model, predict, get_accuracy
from hyperopt import hp, tpe, fmin 
from Score_of_EVM import perf_measure

Xtrain,ytrain = load_data('../letter/train.csv')
Xtest, ytest = load_data('../letter/test.csv')

space = [hp.quniform('tailsize',30,200,5), hp.quniform('cover_threshold',0.2,0.95,0.1),
        hp.quniform('num_to_fuse',1,10,1), hp.quniform('margin_scale',0.3,0.7,0.1) ,
        hp.quniform('ot',0,0.3,0.001)]

def open_set_evm(X_train,y_train,X_test,y_test):
  
    with timer("...fitting train set"):
        weibulls = []
        weibulls = fit(X_train,y_train)
    with timer("...reducing model"):
        X_train,weibulls,y_train = reduce_model(X_train,weibulls,y_train)
    print(("...model size: {}".format(len(y_train))))
    with timer("...getting predictions"):
        predictions,probs = predict(X_test,X_train,weibulls,y_train)
    with timer("...evaluating predictions"):
        accuracy = get_accuracy(predictions,y_test)       
    print("accuracy: {}".format(accuracy))
    return accuracy,predictions, y_test


def tune_func(args):
    global tailsize,cover_threshold,num_to_fuse,margin_scale,ot
    tailsize = int(args[0])
    cover_threshold = args[1]
    num_to_fuse = int(args[2])
    margin_scale = args[3]
    ot = args[4]
    config.ot=ot
    config.tailsize=tailsize
    config.cover_threshold=cover_threshold
    config.num_to_fuse=num_to_fuse
    config.margin_scale=margin_scale
    print(args)
    accuracy, predictions, yactual = open_set_evm(Xtrain,ytrain,Xtest,ytest)
    recognition_accuracy, precision, recall, fmeasure, cm = perf_measure(yactual, predictions)
    print(fmeasure)
    return -fmeasure

if __name__ == '__main__':
    best = fmin(tune_func, space, algo=tpe.suggest, max_evals=200)
    print('Best Parameters obtained are: ', best)