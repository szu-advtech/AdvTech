clc
clear
tic

F_index=1;                     % Problem number
divide_rate=0.3;               % Train and test data rate
Max_Gen=1000;                  % Maximum iteration
ita=0.05;                      % learning rate

[BP] = BP_func(F_index, divide_rate, Max_Gen, ita);%ctrl+D

BP_accuracy_train=BP.accuracy_train;
BP_synapse=BP.synapse;
BP_accuracy_test=BP.accuracy_test;

BP_accuracy_logic_train=BP.accuracy_logic_train;
BP_accuracy_logic_test=BP.accuracy_logic_test;
disp(['修剪后树突分支的个数：',num2str(BP_synapse)]);
disp(['ALNM上训练集的正确率：',num2str(BP_accuracy_train)]);
disp(['ALNM上测试集上的正确率：',num2str(BP_accuracy_test)]);

disp(['逻辑电路上训练集的正确率：',num2str(BP_accuracy_logic_train)]);
disp(['逻辑电路上测试集的正确率：',num2str(BP_accuracy_logic_test)]);

 
 
toc;
%  save BP_accuracy_train BP_accuracy_train 
%  save BP_synapse BP_synapse
%  save BP_accuracy_test BP_accuracy_test
%  save BP_accuracy_logic_train BP_accuracy_logic_train
%  save BP_accuracy_logic_test BP_accuracy_logic_test
 %Over