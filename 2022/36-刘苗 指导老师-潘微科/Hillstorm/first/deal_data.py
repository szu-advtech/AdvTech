import numpy as np
import pandas as pd
from pandas import DataFrame
if __name__ == "__main__":
    data = pd.read_csv('Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv')
    # print(len(data))
    data = data[data['segment']!='Mens E-Mail']
    # print(len(data))
    # print("recency")
    # print(data['recency'].unique())
    print("history_segment")
    print( data['history_segment'].unique())
    # print("history")
    # print( data['history'].unique())
    # print("mens")
    # print( data['mens'].unique())
    # print("womens")
    # print( data['womens'].unique())
    # print("zip_code")
    # print( data['zip_code'].unique())
    print("newbie")
    print( data['newbie'].unique())
    print("channel")
    print( data['channel'].unique())
    print("segment")
    print( data['segment'].unique())

    data = np.array(data)
    # for data_iter in data:
    #     print(data_iter)
    for data_iter in data:
        # print(data_iter)
        if data_iter[1] == '1) $0 - $100':
            data_iter[1] = 1
        elif data_iter[1] == '2) $100 - $200':
            data_iter[1] = 2
        elif data_iter[1] == '3) $200 - $350':
            data_iter[1] = 3
        elif data_iter[1] == '4) $350 - $500':
            data_iter[1] = 4
        elif data_iter[1] == '5) $500 - $750':
            data_iter[1] = 5
        elif data_iter[1] == '6) $750 - $1,000':
            data_iter[1] = 6
        elif data_iter[1] == '7) $1,000 +':
            data_iter[1] = 7
        else:
            print("history_segment")

        if data_iter[5] == 'Surburban':
            data_iter[5] = 1
        elif data_iter[5]== 'Rural':
            data_iter[5] = 2
        elif data_iter[5] == 'Urban':
            data_iter[5] = 3
        else:
            print("newbie")

        if data_iter[7] == 'Phone':
            data_iter[7] = 1
        elif data_iter[7]== 'Web':
            data_iter[7] = 2
        elif data_iter[7] == 'Multichannel':
            data_iter[7] = 3
        else:
            print("channel")

        if data_iter[8] == 'Womens E-Mail':
            data_iter[8] = 1
        elif data_iter[8] == 'No E-Mail':
            data_iter[8] = 0
        else:
            print("sgement")

    data = pd.DataFrame(data)
    data.to_csv('deal_Hillstrom.csv',sep=',',header=True,index=False)