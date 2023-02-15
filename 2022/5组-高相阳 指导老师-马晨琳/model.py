from pmf import *
model = pmf(train_list=train_list,
            test_list=test_list,
            N=N,
            M=M,
            K=K,
            learning_rate=learning_rate,
            lamda_regularizer=lamda_regularizer,
            max_iteration=max_iteration)
P, Q, records_array = model.train()
print('MAE:%.4f;RMSE:%.4f;Recall:%.4f;Precision:%.4f'
      %(records_array[:,1][-1],records_array[:,2][-1],records_array[:,3][-1],records_array[:,4][-1]))

figure(values_list=records_array[:,0],name='loss')
figure(values_list=records_array[:,1],name='MAE')

###############################################################################

from autorec import *
model = autorec(users_num = users_num,
                items_num = items_num,
                hidden_size = hidden_size,
                batch_size = batch_size,
                learning_rate = learning_rate,
                lamda_regularizer = lamda_regularizer)

records_list = []
for epoch in range(epoches):
    data_mat = np.random.permutation(train_mat)
    loss = model.train(data_mat=data_mat)
    pred_mat = model.predict_ratings(data_mat=train_mat)
    mae, rmse, recall, precision = evaluation(pred_mat, train_mat, test_mat)
    records_list.append([loss[-1],mae, rmse, recall, precision])
    if epoch % 10==0:
        print('epoch:%d  loss=%.4f; \n MAE=%.4f; RMSE=%.4f; Recall=%.4f; Precision=%.4f'
              %(epoch, loss[-1], mae, rmse, recall, precision))

figure(values_list=np.array(records_list)[:,0],name='loss')
figure(values_list=np.array(records_list)[:,2],name='RMSE')
figure(values_list=np.array(records_list)[:,-1],name='Precision')