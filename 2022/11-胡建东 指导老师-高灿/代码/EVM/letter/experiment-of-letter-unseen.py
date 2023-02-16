import pandas as pd

from code_with_evm import Code_of_EVM
from code_with_evm import Score_of_EVM




def get_unknow_prob(file):
    #'./test.csv'
    df = pd.read_csv(file, header=None)
    df1 = df[(df.loc[:, 0] == 99)]
    unkown_prob = len(df1) / len(df)
    unkown_prob = format(unkown_prob, '.4f')
    return unkown_prob

def get_unknow(file):
    #'./test.csv'
    df = pd.read_csv(file, header=None)
    df1 = df[(df.loc[:, 0] != 99)]
    unkown_prob = len(df1) / len(df)
    unkown_prob = format(unkown_prob, '.2%')
    return unkown_prob

if __name__ == '__main__':
    #获取f1分数和未知类之间的开放性关系
    df = pd.read_csv('./test.csv', header=None)
    df1 = df[(df.loc[:, 0] != 99)]
    df2=df[(df.loc[:, 0] == 99)]
    df2=df2.reset_index(drop=True)
    # df1.to_csv("./test", header=None, index=None)
    # accuracy, predictions, yactual = EVM.open_set_evm('./train.csv', './test.csv')
    # recognition_accuracy, precision, recall, fmeasure, cm = metrics.perf_measure(yactual, predictions)
    # print(fmeasure)
    inigit=[0.98,0]
    f_probs=[]
    f_probs.append(inigit)
    for i in range(200,len(df2)+1,200):
        df1 = df1.append(df2.loc[i-199:i, :])
        df1.to_csv("./test", header=None, index=None)
        accuracy, predictions, yactual = Code_of_EVM.open_set_evm('./train.csv', './test')
        recognition_accuracy, precision, recall, fmeasure, cm = Score_of_EVM.perf_measure(yactual, predictions)
        prob=get_unknow_prob("./test")
        f_prob=[fmeasure,prob]
        f_probs.append(f_prob)
        print(fmeasure)
    accuracys_probs = pd.DataFrame(data=f_probs)
    accuracys_probs.to_csv('./EVM_of_letter.csv', header=None, index=False)


    # sample_data_prep.unknown_of_letter(0.7)
    #
    # accuracy, predictions, yactual = EVM.open_set_evm('./train.csv', './test.csv')
    # recognition_accuracy, precision, recall, fmeasure, cm = metrics.perf_measure(yactual, predictions)
    #
    # print(
    #     f'Recognition Accuracy: {recognition_accuracy}, F-Measure: {fmeasure}, Precision: {precision}, Recall: {precision}')
    #
    # """
    # #############################################################################################################
    #                             CONFUSION MATRIX
    # #############################################################################################################
    # """
    # for i in range(5,10,1):
    #     sample_data_prep.unknown_of_letter(i*0.1)
    #     print(get_unknow_prob('./test.csv'))
    #
    # A=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    #                  'O','P','Q','R','S','T','U','V','W','X','Y','Z']
    # prob_of_unseen=0
    # accuracys_probs=[]
    # for i in range(len(A)):
    #     accuracy_prob_of_unseen = []
    #     try:
    #         B = A[:(i + 1)]
    #         prob_of_unseen = format(((i + 1) / 26), '.3%')
    #         sample_data_prep.sample_data_letter(B)
    #         accuracy, predictions, yactual = EVM.open_set_evm('train.csv', 'test.csv')
    #         accuracy = format(accuracy, '.3%')
    #         accuracy_prob_of_unseen = [accuracy,prob_of_unseen]
    #         accuracys_probs.append(accuracy_prob_of_unseen)
    #     except :
    #         continue
    # accuracys_probs = pd.DataFrame(data=accuracys_probs)
    # accuracys_probs.to_csv('./EVM_of_letter.csv', header=None, index=False)


    # A = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    #      'O']
    # sample_data_prep.sample_data_letter(A)