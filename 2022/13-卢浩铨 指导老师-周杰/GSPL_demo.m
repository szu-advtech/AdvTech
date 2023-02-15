close all;
clear;
clc;

%% 前沿技术
apram = 0.2;
sign = 'COIL100';
addpath('utils');

%% train/test data
[train_data,train_label,test_data,test_label,data_d,...
        data_n] = func_choDS(apram,'sign',sign);
train_data = double(train_data);
test_data = double(test_data);
train_data = train_data - mean(train_data,2);
test_data = test_data - mean(test_data,2);
% train_data = normalize(train_data,2);
% test_data = normalize(test_data,2);
c = train_label(end);
train_n = length(train_label);
train_onehot = zeros(c,train_n);
for i = 1:train_n
    train_onehot(train_label(i),i) = 1;
end
data = [train_data, test_data];

% view 1 GIST feature
data_v1 = func_formMyGIST(data, 32, 32);
data_v1 = double(data_v1);
train_data_v1 = data_v1(:, 1:train_n);
test_data_v1 = data_v1(:, train_n+1:data_n);

% view 2 LBP feature
data_v2 = func_formMyLBP(data, 32, 32);
data_v2 = double(data_v2);
train_data_v2 = data_v2(:, 1:train_n);
test_data_v2 = data_v2(:, train_n+1:data_n);

% view 3 HOG feature
data_v3 = func_formMyHOG(data, 32, 32);
data_v3 = double(data_v3);
train_data_v3 = data_v3(:, 1:train_n);
test_data_v3 = data_v3(:, train_n+1:data_n);

train_data_w(3) = struct('X_v', [], 'dv', []);
train_data_w(1).X_v = train_data_v1';
[train_data_w(1).dv, ~] = size(train_data_v1);
train_data_w(2).X_v = train_data_v2';
[train_data_w(2).dv, ~] = size(train_data_v2);
train_data_w(3).X_v = train_data_v3';
[train_data_w(3).dv, ~] = size(train_data_v3);
% train_data_w(4).X_v = train_data';
% [train_data_w(4).dv, ~] = size(train_data);

[W, p, z] = func_GSPL_adv(train_data_w, train_label, train_n, 3, 100, 1e3, 268, 100, 10, 10);
gspl_label = func_kNN(W' * [test_data_v1; test_data_v2; test_data_v3],...
    W' * [train_data_v1; train_data_v2; train_data_v3], train_label, 3);
acc_gspl = func_getRecogAcc(test_label, gspl_label);

gspl_label_v1 = func_kNN(W(1:512, :)' * test_data_v1,...
    W(1:512, :)' * train_data_v1, train_label, 3);
acc_gspl_v1 = func_getRecogAcc(test_label, gspl_label_v1);

gspl_label_v2 = func_kNN(W(513:571, :)' * test_data_v2,...
    W(513:571, :)' * train_data_v2, train_label, 3);
acc_gspl_v2 = func_getRecogAcc(test_label, gspl_label_v2);

gspl_label_v3 = func_kNN(W(572:895, :)' * test_data_v3,...
    W(572:895, :)' * train_data_v3, train_label, 3);
acc_gspl_v3 = func_getRecogAcc(test_label, gspl_label_v3);
