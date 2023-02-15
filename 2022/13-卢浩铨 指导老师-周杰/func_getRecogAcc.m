function [acc] = func_getRecogAcc(test_label, re_label)
%% Accuracy calculator for feature extraction
% input test_label : 1*n;
% output acc : acc rate.

%% function body
    right_num = 0;
    test_num = size(test_label);
    test_num = test_num(2);
    for i =1:test_num
        if test_label(i) == re_label(i)
            right_num = right_num + 1;
        end
    end
    acc = right_num / test_num * 100;
end