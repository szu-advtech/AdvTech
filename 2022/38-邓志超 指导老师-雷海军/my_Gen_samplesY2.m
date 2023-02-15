
% ��֮ǰ������������ͬ��������������û�н���ת�ã��б�ʾ��ͬ���������б�ʾ����������������

function [Xtrain,Ltrain,Xtest,Ltest,Atrain,Atest,Btrain,Btest,Ytrain] = ...
    my_Gen_samplesY2(Xpar1,Xpar2,posY,negY,posScoresA,negScoresA,posScoresB,negScoresB,labelPar1,labelPar2,indPar1,indPar2,i)
% Yʵ�����Ǿ�������Ĺ��� �÷ֺͱ�ǩ.
% separately sampling from each class


% ������
test1 = (indPar1==i);      % ����Ϊ1�Ĳ���
train1 = ~test1;        % ȡ��

% ����������Ϊ i �Ĳ��֣���������
Xtest1 = Xpar1(test1,:);
Ltest1 = labelPar1(test1,:);
posAtest = posScoresA(test1,:);
posBtest = posScoresB(test1,:);

% ������ȡ��i�Ĳ��֣�����ѵ��
Xtrain1 = Xpar1(train1,:);     % fea * ins
Ltrain1 = labelPar1(train1,:);	% label*1
posYtrain = posY(train1,:);     % score * ins
posAtrain = posScoresA(train1,:);
posBtrain = posScoresB(train1,:);

% ������
test2 = (indPar2 ==i);    
train2 = ~test2;

% ����������Ϊ i �Ĳ���
Xtest2 = Xpar2(test2,:);
Ltest2 = labelPar2(test2,:);
negAtest = negScoresA(test2,:);
negBtest = negScoresB(test2,:);

% ������ȡ�� i �Ĳ���
Xtrain2 = Xpar2(train2,:);     %% fea * ins
Ltrain2 = labelPar2(train2,:);	%% label*1
negYtrain = negY(train2,:);     %% score * ins
negAtrain = negScoresA(train2,:);
negBtrain = negScoresB(train2,:);

%final results
Xtrain = [Xtrain1;Xtrain2];     % ѵ�������� �����������͸���������Ӧ�ı�ǩ�Լ��÷�������
Ltrain = [Ltrain1;Ltrain2];     % ѵ���ı�ǩ �����������͸�����
Ytrain = [posYtrain;negYtrain]; % ѵ���Ĵ����ĵ÷����� �����������͸�����
Atrain = [posAtrain;negAtrain]; % ѵ���� scoresA �÷� �����������͸�����
Btrain = [posBtrain;negBtrain]; % ѵ���� scoresB �÷� �����������͸�����

Xtest = [Xtest1;Xtest2];        % ���Ե����� �������У���Ӧ�ı�ǩ�Լ��÷�������
Ltest = [Ltest1;Ltest2];        % ���Եı�ǩ
Atest = [posAtest;negAtest];    % ���Ե� scoresA �÷�
Btest = [posBtest;negBtest];    % ���Ե� scoresB �÷�


% Ytest = [Ytest1 Ytest2];
