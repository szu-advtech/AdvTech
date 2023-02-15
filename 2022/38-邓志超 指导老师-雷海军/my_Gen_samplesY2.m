
% 与之前的样本产生不同，这里的输入矩阵，没有进行转置，行表示不同的样本，列表示脑区（特征）数。

function [Xtrain,Ltrain,Xtest,Ltest,Atrain,Atest,Btrain,Btest,Ytrain] = ...
    my_Gen_samplesY2(Xpar1,Xpar2,posY,negY,posScoresA,negScoresA,posScoresB,negScoresB,labelPar1,labelPar2,indPar1,indPar2,i)
% Y实际上是经过处理的过的 得分和标签.
% separately sampling from each class


% 正样本
test1 = (indPar1==i);      % 划分为1的部分
train1 = ~test1;        % 取非

% 正样本划分为 i 的部分，用来测试
Xtest1 = Xpar1(test1,:);
Ltest1 = labelPar1(test1,:);
posAtest = posScoresA(test1,:);
posBtest = posScoresB(test1,:);

% 正样本取非i的部分，用来训练
Xtrain1 = Xpar1(train1,:);     % fea * ins
Ltrain1 = labelPar1(train1,:);	% label*1
posYtrain = posY(train1,:);     % score * ins
posAtrain = posScoresA(train1,:);
posBtrain = posScoresB(train1,:);

% 负样本
test2 = (indPar2 ==i);    
train2 = ~test2;

% 负样本划分为 i 的部分
Xtest2 = Xpar2(test2,:);
Ltest2 = labelPar2(test2,:);
negAtest = negScoresA(test2,:);
negBtest = negScoresB(test2,:);

% 负样本取非 i 的部分
Xtrain2 = Xpar2(train2,:);     %% fea * ins
Ltrain2 = labelPar2(train2,:);	%% label*1
negYtrain = negY(train2,:);     %% score * ins
negAtrain = negScoresA(train2,:);
negBtrain = negScoresB(train2,:);

%final results
Xtrain = [Xtrain1;Xtrain2];     % 训练的样本 包括正样本和负样本，对应的标签以及得分在下面
Ltrain = [Ltrain1;Ltrain2];     % 训练的标签 包括正样本和负样本
Ytrain = [posYtrain;negYtrain]; % 训练的处理后的得分样本 包括正样本和负样本
Atrain = [posAtrain;negAtrain]; % 训练的 scoresA 得分 包括正样本和负样本
Btrain = [posBtrain;negBtrain]; % 训练的 scoresB 得分 包括正样本和负样本

Xtest = [Xtest1;Xtest2];        % 测试的样本 正负都有，对应的标签以及得分在下面
Ltest = [Ltest1;Ltest2];        % 测试的标签
Atest = [posAtest;negAtest];    % 测试的 scoresA 得分
Btest = [posBtest;negBtest];    % 测试的 scoresB 得分


% Ytest = [Ytest1 Ytest2];
