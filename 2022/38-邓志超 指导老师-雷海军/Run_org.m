

% Com_SVM_SVR_new(NewXTrain,NewXTest,Mtrain,Mtest,Ltrain,Ltest); 
% 116����   +  116����   +	116DTI   +   CSF   +   [depScores,sleepScores,smellScores,MoCASores]   +   label  +  116CSF  = 472
% [1:116]     [117:232]    [233:348]  [349:351]             [352:355]                                  [356]    [357:472]= 472

clear 

addpath(genpath('d:\�ҵ��ĵ�\MATLAB\Parkinson\libsvm-mat-2.91-1'));
% addpath(genpath('..\'));

% load PPMI_T1_T1_DTI_CSF_DSSM_Label_208;
load PPMI_T1G_T1W_DTI_CSF_DSSM_Label_T1CSF_208;
dataT1_CSF_DTI=PPMI_T1G_T1W_DTI_CSF_DSSM_Label_CSF;

T1_G_116=dataT1_CSF_DTI(:,1:116);   % 116 ����
T1_W_116=dataT1_CSF_DTI(:,117:232); % 116 ����
DTI_116=dataT1_CSF_DTI(:,233:348);  % 116 DTI
CSF=dataT1_CSF_DTI(:,349:351);      % 3 CSF
DSSM=dataT1_CSF_DTI(:,352:355);     % 4 �÷�
label=dataT1_CSF_DTI(:,356);        % 1 ��ǩ
T1CSF=dataT1_CSF_DTI(:,357:472);    % 116 T1_CSF

isNC=(label==1);            % NC=56 PD=123 SWEDD=29
isPD=(label==-1);
isSWEDD=(label==2);
    
SELECT={'isNC','isPD','NC','PD'};           % ѡ��ͬ�����ʵ�� NCvsPD NCvsSWEDD PDvsSWEDD
% SELECT={'isNC','isSWEDD','NC','SWEDD'};
% SELECT={'isPD','isSWEDD','PD','SWEDD'};

IndexStirng=['Index_',SELECT{3},'_vs_',SELECT{4}];
load (IndexStirng); % ��ȡ������֤�ķ����ǩ��֮ǰ�Ѿ�������˵ı�ǩ��

D=[T1_G_116,T1_W_116,DTI_116,CSF,DSSM,label,T1CSF];  % �ܹ���6����ɲ��֣����ʡ����ʡ�DTI��CSF��DSSM��label
COMPOSE1=D(eval(SELECT{1}),:);     % NC=56 AD=123 SWEDD=29
COMPOSE2=D(eval(SELECT{2}),:);     % NC=56 AD=123 SWEDD=29      

data=[COMPOSE1;COMPOSE2];


data_T1_G_116=data(:,1:10:116);     % 116 T1
data_T1_W_116=data(:,117:232);      % 116 T1
data_DTI_116=data(:,233:348);       % 116 DTI
data_CSF=data(:,349:351);           % 3 CSF
data_T1CSF=data(:,357:472);         % 116 T1CSF

% �ع����YSocre
data_depScores=data(:,352);
data_sleepScores=data(:,353);
data_smellScores=data(:,354);
data_MoCASores=data(:,355);
data_label=data(:,356);
data_YScore=[data_sleepScores,data_smellScores,data_label];         % sleepScores,smellScores label

X1=data_T1_G_116;   
X1=my_convert2Sparse(X1); % ÿ��ֵ����ֵ�����Ա�׼��.����ת����sparse����

X2=data_T1_W_116;  
X2=my_convert2Sparse(X2); % ÿ��ֵ����ֵ�����Ա�׼��.����ת����sparse����

Y=data_DTI_116;  
Y=my_convert2Sparse(Y); % ÿ��ֵ����ֵ�����Ա�׼��.����ת����sparse����

Z=data_CSF;
Z=my_convert2Sparse(Z); % ÿ��ֵ����ֵ�����Ա�׼��.����ת����sparse����

% �ع����
L=data_YScore;
L=my_convert2Sparse(L); % ÿ��ֵ����ֵ�����Ա�׼��.����ת����sparse����

% T1�Լ�ҺCSF
M=data_T1CSF;  
M=my_convert2Sparse(M); % ÿ��ֵ����ֵ�����Ա�׼��.����ת����sparse����


flagCOMPOSE1=ones(size(COMPOSE1,1),1);        % COMPOSE1 Ϊ +1
flagCOMPOSE2=-1*ones(size(COMPOSE2,1),1);     % COMPOSE2 Ϊ -1
mergeflag=[flagCOMPOSE1;flagCOMPOSE2];  % ʵ����ǰ���data_label���
% mergeData=X1;
% mergeData=X2;.
% mergeData=M;    % 116CSF
% mergeData=[X1,M];    % 116CSF
% mergeData=[X1,Y,Z];
% mergeData=[X1,X2,Y,Z];
% mergeData=[X1,Y,Z,M];       % X1������   X2������   %Y��DTI    %Z����CSF    %M:T1CSF
mergeYScore=L;              % L���ٴ��÷�


kFold= 10; % ���������� k �λ���
pars.k1 = 10; % ÿ�λ��ֵ��Ӽ����� kFold

parc = -5:1:5; % libsvm�Ĳ������ã�2��-5�η���>2��5�η�
parg = -5:1:5;
% parc = 2; % libsvm�Ĳ������ã�2��-5�η���>2��5�η�
% parg = -2;

% parc = -5:1:10; % libsvm�Ĳ������ã�2��-5�η���>2��5�η�
% parg = -5:1:5;
% parc = -2:1:2; % libsvm�Ĳ������ã�2��-5�η���>2��5�η�
% parg = -2:1:2;
% 
pars.lambda1 = 1;
pars.lambda2 = 1;
pars.lambda3 = 1;

% t1=-5:2;
% t2=-5:2;
% t3=2:8;
% pars.lambda1 = 10.^t1;
% pars.lambda2 = 10.^t2;
% pars.lambda3 = 10.^t3;

pars.Ite=50;

% ǰ���Ѿ��ѱ����˷����ǩ��ȡ��
for k = 1:kFold  % ��������Ŀ������10�� 10-Kfold ����
    ind1(:,k) = crossvalind('Kfold',size(COMPOSE1,1),pars.k1); % ��  COMPOSE1 ��ǩ��������ֳ�10��
    ind2(:,k) = crossvalind('Kfold',size(COMPOSE2,1),pars.k1); % ��  COMPOSE2 ��ǩ��������ֳ�10��
end

posSampleNum=length(flagCOMPOSE1);

Xpar1=mergeData(1:posSampleNum,:);      % ������
Xpar2=mergeData(posSampleNum+1:end,:);	% ������

Ypar1=mergeYScore(1:posSampleNum,:);        % ������ �ع���
Ypar2=mergeYScore(posSampleNum+1:end,:);	% ������ �ع���

% data_depScores  data_sleepScores   data_smellScores  data_MoCASores
scoresApar1=data_sleepScores(1:posSampleNum,:);
scoresApar2=data_sleepScores(posSampleNum+1:end,:);	
scoresBpar1=data_smellScores(1:posSampleNum,:);
scoresBpar2=data_smellScores(posSampleNum+1:end,:);

labelPar1=flagCOMPOSE1;           % ������ ��ǩ������ȫ�ֱ����Ķ���,Ϊ�˲��е�ʱ������.
labelPar2=flagCOMPOSE2;           % ������ ��ǩ�� 


tic

len3=length(pars.lambda3);  % ����ȫ�ֱ����Ķ���,Ϊ�˲��е�ʱ������.
len2=length(pars.lambda2);
len1=length(pars.lambda1);
TOTAL=len1*len2*len3;   % �ܹ���Ҫѭ���Ĵ���

SelectFeaIdx=cell(kFold,pars.k1);   % SelectFeaIdx ��10��10���ֽ�����֤��ѡ��������±�.10*10ά��
W=cell(kFold,pars.k1);              % W ��10��10���ֽ�����֤ѭ���еĻع�ϵ��.10*10ά��
Ite=pars.Ite;                       % ע��,����ǵ�������.Ҳ��Ϊ������.


Res1=my_initialRes; % ��ʼ��
Res2=my_initialRes; % ��ʼ��


for l3 = 1:len3 % pars.lambda3 = 1
    lamb3=pars.lambda3(l3); 
    for l2 = 1:len2 % pars.lambda2 = 1
        lamb2=pars.lambda2(l2); 
        for l1 = 1:len1 % pars.lambda1 = 1  
            lamb1=pars.lambda1(l1); % ��������lambda��ѭ����ʼ�Ͷ�����,Ҳ��Ϊ������.
            
            singleStartTime=toc;  % ��¼ÿ��ѭ���Ŀ�ʼʱ��(��ײ��ѭ��)            
            hasFinished=(l3-1)*len2*len1+(l2-1)*len1+l1; % ͳ������forѭ����ɸ���
            fprintf('Doing��l3=%d/%d l2=%d/%d l1=%d/%d.\nAfter this will finish:%.2f%%(%d/%d)\n',l3,len3,l2,len2,l1,len1,hasFinished/TOTAL*100,hasFinished,TOTAL);
          
            for kk = 1:kFold % ��ʾ��10�� 10-Kfold ����
                indPar1=ind1(:,kk);
                indPar2=ind2(:,kk);
                
                parfor ii = 1:pars.k1 % pars.k1 = 10��ÿ�λ���Ϊ10�飬��ʾ10��ѭ���Ľ�����֤
%                 for ii = 1:2 % ֻ��һ�εĽ��
                    % ����ѵ������ ��������
                    [trainData,trainFlag,testData,testFlag,Atrain,Atest,Btrain,Btest,Ytrain] = my_Gen_samplesY2(...
                        Xpar1,Xpar2,...             % X ������/������
                        Ypar1,Ypar2,...             % ������/������ ������� score��label
                        scoresApar1,scoresApar2,...	% ������/������ scoresA
                        scoresBpar1,scoresBpar2,...	% ������/������ scoresB
                        labelPar1,labelPar2,...     % ������/������ label
                        indPar1,indPar2,...         % ��k�ε� ������/����������
                        ii);
                    
%                     1. generate regression coefficient
                    
                    %  ������������ ��Ҫ��ԭ�������ת��
%                     warning('off');
%                     W{kk,ii} = LF3L21(trainData',Ytrain',lamb1,lamb2,lamb3,Ite);    
%                     
%                     % 2. feature selection
%                     normW = sqrt(sum(W{kk,ii}.*W{kk,ii},2)); % ������ͣ�ȡ����
%                     normW( normW <= 0.2 * mean(normW) )=0; % ѡȡ����ص���������������0�������ѡȡ�������Ը���.
%                     SelectFeaIdx{kk,ii} = find(normW~=0); % SelectFeaIdx: ����ص������±�
%                     newTrainData = trainData(:,SelectFeaIdx{kk,ii}); % ѡȡ����ص�����ά��
%                     newTestData = testData(:,SelectFeaIdx{kk,ii});

                    newTrainData=trainData;
                    newTestData=testData;
                    
                    % ע��auc��rmse���Ǹ��Է��ص���parc pargѭ���е����ֵ���� SVMacc��Acc��Bcc�޹�ϵ.
                    [fsAcc(kk,ii),fsSen(kk,ii),fsSpec(kk,ii),fsPrec(kk,ii),maxFscore(kk,ii),maxAuc1(kk,ii),...
                        maxAcc1(kk,ii),minArmse1(kk,ii),maxBcc1(kk,ii),minBrmse1(kk,ii),...
                        Rs1{kk,ii},Ra1{kk,ii},Rb1{kk,ii}]=my_combineSVM_SVR_maxFS(newTrainData,trainFlag,Atrain,Btrain,newTestData,testFlag,Atest,Btest,parc,parg);
                    
                    [svmMaxAcc(kk,ii),accSen(kk,ii),accSpec(kk,ii),accPrec(kk,ii),accFscore(kk,ii),maxAuc2(kk,ii),...
                        maxAcc2(kk,ii),minArmse2(kk,ii),maxBcc2(kk,ii),minBrmse2(kk,ii),...
                        Rs2{kk,ii},Ra2{kk,ii},Rb2{kk,ii}]=my_combineSVM_SVR_maxAcc(newTrainData,trainFlag,Atrain,Btrain,newTestData,testFlag,Atest,Btest,parc,parg);

                end
            end            
            
            singleEndTime=toc;    % ����ÿ��ѭ�����õ�ʱ��(��ײ�)
            fprintf('����ѭ�����õ�ʱ��: %f\n',singleEndTime-singleStartTime);
%             ��¼ÿһ�ν�����֤��ƽ��ֵ
            Res1=my_updateRes(Res1,fsAcc,fsSen,fsSpec,fsPrec,maxFscore,maxAuc1,maxAcc1,minArmse1,maxBcc1,minBrmse1,Rs1,Ra1,Rb1,l1,l2,l3,SelectFeaIdx,W);
            Res2=my_updateRes(Res2,svmMaxAcc,accSen,accSpec,accPrec,accFscore,maxAuc2,maxAcc2,minArmse2,maxBcc2,minBrmse2,Rs2,Ra2,Rb2,l1,l2,l3,SelectFeaIdx,W);
        end
    end
end



toc
endTime=toc;
fprintf('%fs\n',endTime);
fprintf('%s_vs_%s\n',SELECT{3},SELECT{4});

save test.mat Res1 Res2
% clear Res
% Res=Res_1;
% Res=Res_2;


%    SVM for classification on class label
% function [a,b,c,d,e,f,g,h,i,j,k,AUC,R] = Com_SVM_SVR_new(NewXTrain,NewXTest,Atrain,Atest,Mtrain,Mtest,Ltrain,Ltest)
% cc:���׼ȷ��(ƽ�����ϵ��)
% rmse:��С�������
% SVM.acc:  accuracy ׼ȷ��
% SVM.sen:  sensitivity �����ԡ�����⵽����������Ŀռ����ǩ��Ŀ�ı������е�����R���ٻ��ʣ�
% SVM.spec: specificity �����ԡ�����⵽����������Ŀռ����ǩ��Ŀ�ı�������Ϊ -R
% SVM.Fscore: fscore f-measure: 2PR/(P+R)
% SVM.Prec:	precision ���� 
% SVM.auc:	AUC���ֵ




