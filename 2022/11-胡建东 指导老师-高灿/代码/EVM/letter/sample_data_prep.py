import pandas as pd


import os
def unknown_of_letter(frac):
    print(os.getcwd())  # 看相对路径

    df = pd.read_csv('letter-recognition.csv', header=None)

    #数据分割

    train = df.sample(frac=frac, random_state=200)  # considering 70% of the data as train data and 30% as test
    test = df.drop(train.index)

    #训练数据
    A = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O']
    known_classes = A  # comsidering 15 classses as known classes  'P','Q','R','S','T'
    target_known_df = train.loc[df.iloc[:, 0].isin(known_classes)]
    target_known_df = target_known_df.sort_values(0)
    target_known_df = target_known_df.replace(known_classes,
                                              list(range(1, 1 + len(known_classes))))

    #测试数据获取
    test = test.sort_values(0)
    test_classes = test[0].unique().tolist()
    unknown_classes = list(set(test_classes) - set(known_classes))
    test = test.replace(unknown_classes, [99] * len(unknown_classes))
    test = test.replace(known_classes, list(range(1, 1 + len(known_classes))))


    #保存数据到文本
    target_known_df.to_csv('./train.csv', header=None, index=False)
    test.to_csv('./test.csv', header=None, index=False)
