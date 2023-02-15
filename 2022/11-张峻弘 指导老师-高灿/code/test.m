clc;
clear;
close;

%% MACRO SETTING 
PARENT_PATH = "../";
addpath("utils/")

%% Experiment setting
trainRate = 0.6;       % 60% of the samples will be used for training.
dataFolder = strcat(PARENT_PATH, "data/");
dataName = 'dna_scale.mat';
data = load(dataFolder + dataName);

rng(1024); % set RNG

%% training/testing data splitting
n = size(data.sample, 1);
nTr = round(n*trainRate);

train_ind = false(n, 1);
train_ind(randperm(n, nTr)) = true;

%% Method setting
Nw  = 5000;
rho = 1.024;
tol = 1e-8;

param.dim = Nw;
param.sigma = 8;
param.rho = rho;
param.tol = tol;

Xtrain = data.sample(train_ind, :);
Xtest = data.sample(~train_ind, :);
Ytrain = data.label(train_ind);
Ytest = data.label(~train_ind);

disp("== rff-svm ==");
model = rff_svm(Xtrain, Ytrain, param, "featureType", "rff", "sign", true);
Ztest = makeRFF(model.rffW, model.rffb, Xtest');
Ztest = sign(Ztest);
[pred, acc, ~]= liblinearpredict(Ytest, sparse(Ztest'), model.svm);

disp("== rmbcsik-svm ==");
model = rff_svm(Xtrain, Ytrain, param, "featureType", "rm-bcsik", "sign", true);
Ztest = makeRFF(model.rffW, model.rffb, Xtest');
Ztest = sign(Ztest);
[pred, acc, ~]= liblinearpredict(Ytest, sparse(Ztest'), model.svm);

disp("== improved-bcsik-svm ==");
model = rff_svm(Xtrain, Ytrain, param, "featureType", "optimized", "sign", true);
Ztest = makeRFF(model.rffW, model.rffb, Xtest');
Ztest = sign(Ztest);
[pred, acc, ~] = liblinearpredict(Ytest, sparse(Ztest'), model.svm);

disp("== exact-kernel-svm ==");
svm = libsvmtrain(Ytrain, Xtrain, '-g 0.01 -q');
[pred, acc, ~] = libsvmpredict(Ytest, Xtest, svm);