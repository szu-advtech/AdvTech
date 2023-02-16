function [BP] = BP_func(F_index, divide_rate, epoch, ita)
[ input_train, target_train, input_test, target_test, M, PS ] = divideDataset( F_index, divide_rate);

% training the trained dat
..............................................................................................................................................................................................................................a
net=newnanm(input_train,target_train,ita,M,epoch);              

%% Evaluate the accuracy of ALNM
[BP.accuracy_train, BP.synapse] = evaluate_accuracy( input_train, target_train, net, M );
[BP.accuracy_test, ~] = evaluate_accuracy( input_test, target_test, net, M );

%% Evaluate the accuracy of the logic circuit classifier
[BP.accuracy_logic_train, BP.Threshold_logic] = evaluate_accuracy_logic_circuit( input_train, target_train,  net, M, PS);
[BP.accuracy_logic_test, ~] = evaluate_accuracy_logic_circuit( input_test, target_test,  net, M, PS);
end
% Over


