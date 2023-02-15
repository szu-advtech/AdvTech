

% Com_SVM_SVR_new(NewXTrain,NewXTest,Mtrain,Mtest,Ltrain,Ltest); 
% 116灰质   +  116白质   +	116DTI   +   CSF   +   [depScores,sleepScores,smellScores,MoCASores]   +   label  +  116CSF  = 472
% [1:116]     [117:232]    [233:348]  [349:351]             [352:355]                                  [356]    [357:472]= 472

clear 

addpath(genpath('d:\我的文档\MATLAB\Parkinson\libsvm-mat-2.91-1'));

% load PPMI_T1_T1_DTI_CSF_DSSM_Label_208;
load PPMI_T1G_T1W_DTI_CSF_DSSM_Label_T1CSF_208;
dataT1_CSF_DTI=PPMI_T1G_T1W_DTI_CSF_DSSM_Label_CSF;

T1_G_116=dataT1_CSF_DTI(:,1:116);   % 116 灰质
T1_W_116=dataT1_CSF_DTI(:,117:232); % 116 白质
DTI_116=dataT1_CSF_DTI(:,233:348);  % 116 DTI
CSF=dataT1_CSF_DTI(:,349:351);      % 3 CSF
DSSM=dataT1_CSF_DTI(:,352:355);     % 4 得分
label=dataT1_CSF_DTI(:,356);        % 1 标签
T1CSF=dataT1_CSF_DTI(:,357:472);    % 116 T1_CSF

isNC=(label==1);
isPD=(label==-1);
isSWEDD=(label==2);
isPDSW=(label==-1|label==2);    % PD + SWEDD

D=[T1_G_116,T1_W_116,DTI_116,CSF,DSSM,label,T1CSF];  % 总共有6个组成部分：灰质、白质、DTI、CSF、DSSM、label
NC=D(isNC,:);           % 56
PD=D(isPD,:);           % 123
SWEDD=D(isSWEDD,:);     % 29
PDSW=D(isPDSW,:);

% 一、选择NCvsPD
data=[NC;PD];
flagNC=ones(size(NC,1),1);      % NC 为 +1
flagPD=-1*ones(size(PD,1),1);	% AD 为 -1
mergeflag=[flagNC;flagPD];  % 实际与前面的data_label相等


% 二、选择 NCvsSWEDD
% data=[NC;SWEDD];
% flagNC=ones(size(NC,1),1);      % NC 为 +1
% flagSWEDD=-1*ones(size(SWEDD,1),1);	% AD 为 -1
% mergeflag=[flagNC;flagSWEDD];  % 实际与前面的data_label相等


data_T1_G_116=data(:,1:116);        % 116 T1
data_T1_W_116=data(:,117:232);      % 116 T1
data_DTI_116=data(:,233:348);       % 116 DTI
data_CSF=data(:,349:351);           % 3 CSF
data_T1CSF=data(:,357:472);         % 116 T1CSF

% 回归变量YSocre
data_depScores=data(:,352);
data_sleepScores=data(:,353);
data_smellScores=data(:,354);
data_MoCASores=data(:,355);
data_label=data(:,356);
data_YScore=[data_sleepScores,data_smellScores,data_label];         % sleepScores,smellScores label

X1=data_T1_G_116;   
X1=my_convert2Sparse(X1); % 每个值减均值，除以标准差.并且转换成sparse矩阵

X2=data_T1_W_116;
% X2=my_convert2Sparse(X2); % 每个值减均值，除以标准差.并且转换成sparse矩阵

Y=data_DTI_116;
Y=my_convert2Sparse(Y); % 每个值减均值，除以标准差.并且转换成sparse矩阵

Z=data_CSF;
Z=my_convert2Sparse(Z); % 每个值减均值，除以标准差.并且转换成sparse矩阵

% 回归变量
L=data_YScore;
L=my_convert2Sparse(L); % 每个值减均值，除以标准差.并且转换成sparse矩阵

% T1脑脊液CSF
M=data_T1CSF;
M=my_convert2Sparse(M); % 每个值减均值，除以标准差.并且转换成sparse矩阵

% mergeData=X1;
% mergeData=X2;.
% mergeData=M;              % 116CSF
% mergeData=[X1,M];         % 116CSF
% mergeData=[X1,Y,Z];
% mergeData=[X1,X2,Y,Z];
mergeData=[X1,Y,Z,M];       % X1：灰质   X2：白质   %Y：DTI    %Z生物CSF    %M:T1CSF
mergeYScore=L;              % L：临床得分

kFold= 1; % 对样本进行 k 次划分
pars.k1 = 10; % 每次划分的子集个数 kFold

% parc = -5:1:5; % libsvm的参数设置：2的-5次方—>2的5次方
% parg = -5:1:5;
% parc = 2; % libsvm的参数设置：2的-5次方—>2的5次方
% parg = -2;

% parc = -5:1:10; % libsvm的参数设置：2的-5次方—>2的5次方
% parg = -5:1:5;
parc = -2:1:2; % libsvm的参数设置：2的-5次方—>2的5次方
parg = -2:1:2;

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


for k = 1:kFold  % 对样本数目，进行10次 10-Kfold 划分
    ind1(:,k) = crossvalind('Kfold',size(find(label ==1),1),pars.k1); % 将  NC 标签均匀随机分成10组
    ind2(:,k) = crossvalind('Kfold',size(find(label ==-1),1),pars.k1); % 将  AD 标签均匀随机分成10组
end


posSampleNum=length(flagNC);

Xpar1=mergeData(1:posSampleNum,:);      % 正样本
Xpar2=mergeData(posSampleNum+1:end,:);	% 负样本

Ypar1=mergeYScore(1:posSampleNum,:);        % 正样本 回归量
Ypar2=mergeYScore(posSampleNum+1:end,:);	% 负样本 回归量

% data_depScores  data_sleepScores   data_smellScores  data_MoCASores
scoresApar1=data_sleepScores(1:posSampleNum,:);
scoresApar2=data_sleepScores(posSampleNum+1:end,:);	
scoresBpar1=data_smellScores(1:posSampleNum,:);
scoresBpar2=data_smellScores(posSampleNum+1:end,:);

labelPar1=flagNC;           % 正样本 标签：类似全局变量的定义,为了并行的时候提速.
labelPar2=flagPD;           % 负样本 标签： 


tic

len3=length(pars.lambda3);  % 类似全局变量的定义,为了并行的时候提速.
len2=length(pars.lambda2);
len1=length(pars.lambda1);
TOTAL=len1*len2*len3;   % 总共需要循环的次数

SelectFeaIdx=cell(kFold,pars.k1);   % SelectFeaIdx 是10次10划分交叉验证所选择的脑区下标.10*10维度
W=cell(kFold,pars.k1);              % W 是10次10划分交叉验证循环中的回归系数.10*10维度
Ite=pars.Ite;                       % 注意,这个是迭代次数.也是为了提速.


Res1=my_initialRes; % 初始化
Res2=my_initialRes; % 初始化




for l3 = 1:len3 % pars.lambda3 = 1
    lamb3=pars.lambda3(l3); 
    for l2 = 1:len2 % pars.lambda2 = 1
        lamb2=pars.lambda2(l2); 
        for l1 = 1:len1 % pars.lambda1 = 1  
            lamb1=pars.lambda1(l1); % 其他两个lambda在循环开始就定义了,也是为了提速.
            
            singleStartTime=toc;  % 记录每次循环的开始时间(最底层的循环)            
            hasFinished=(l3-1)*len2*len1+(l2-1)*len1+l1; % 统计三层for循环完成个数
            fprintf('Doing：l3=%d/%d l2=%d/%d l1=%d/%d.\nAfter this will finish:%.2f%%(%d/%d)\n',l3,len3,l2,len2,l1,len1,hasFinished/TOTAL*100,hasFinished,TOTAL);
          
            for kk = 1:kFold % 表示：10次 10-Kfold 划分
                indPar1=ind1(:,kk);
                indPar2=ind2(:,kk);
                
%                 parfor ii = 1:pars.k1 % pars.k1 = 10，每次划分为10组，表示10次循环的交叉验证
                for ii = 1:2 % 只看一次的结果
                    % 产生训练样本 测试样本
                    [trainData,trainFlag,testData,testFlag,Atrain,Atest,Btrain,Btest,Ytrain] = my_Gen_samplesY2(...
                        Xpar1,Xpar2,...             % X 正样本/负样本
                        Ypar1,Ypar2,...             % 正样本/负样本 处理过的 score、label
                        scoresApar1,scoresApar2,...	% 正样本/负样本 scoresA
                        scoresBpar1,scoresBpar2,...	% 正样本/负样本 scoresB
                        labelPar1,labelPar2,...     % 正样本/负样本 label
                        indPar1,indPar2,...         % 第k次的 正样本/负样本划分
                        ii);
                    
%                     1. generate regression coefficient
                    
                    % 下面的这个计算 需要将原矩阵进行转置
%                     warning('off');
%                     W{kk,ii} = LF3L21(trainData',Ytrain',lamb1,lamb2,lamb3,Ite);    
%                     
%                     %2. feature selection
%                     normW = sqrt(sum(W{kk,ii}.*W{kk,ii},2)); % 对行求和，取根号
%                     normW( normW <= 10^-13 * mean(normW) )=0; % 选取最相关的特征，其他的置0，这里的选取方法可以更改.
%                     SelectFeaIdx{kk,ii} = find(normW~=0); % SelectFeaIdx: 最相关的特征下标
%                     newTrainData = trainData(:,SelectFeaIdx{kk,ii}); % 选取最相关的特征维度
%                     newTestData = testData(:,SelectFeaIdx{kk,ii});

                    newTrainData=trainData;
                    newTestData=testData;
                    
                    % 注意auc、rmse都是各自返回的是parc parg循环中的最大值，与 SVMacc、Acc、Bcc无关系.
                    [fsAcc(kk,ii),fsSen(kk,ii),fsSpec(kk,ii),fsPrec(kk,ii),maxFscore(kk,ii),maxAuc1(kk,ii),...
                        maxAcc1(kk,ii),minArmse1(kk,ii),maxBcc1(kk,ii),minBrmse1(kk,ii),...
                        Rs1{kk,ii},Ra1{kk,ii},Rb1{kk,ii}]=my_combineSVM_SVR_maxFS(newTrainData,trainFlag,Atrain,Btrain,newTestData,testFlag,Atest,Btest,parc,parg);
                    
                    [svmMaxAcc(kk,ii),accSen(kk,ii),accSpec(kk,ii),accPrec(kk,ii),accFscore(kk,ii),maxAuc2(kk,ii),...
                        maxAcc2(kk,ii),minArmse2(kk,ii),maxBcc2(kk,ii),minBrmse2(kk,ii),...
                        Rs2{kk,ii},Ra2{kk,ii},Rb2{kk,ii}]=my_combineSVM_SVR_maxAcc(newTrainData,trainFlag,Atrain,Btrain,newTestData,testFlag,Atest,Btest,parc,parg);

                end
            end            
%             记录每一次交叉验证的平均值
            Res1=my_updateRes(Res1,fsAcc,fsSen,fsSpec,fsPrec,maxFscore,maxAuc1,maxAcc1,minArmse1,maxBcc1,minBrmse1,Rs1,Ra1,Rb1,l1,l2,l3,SelectFeaIdx);
            Res2=my_updateRes(Res2,svmMaxAcc,accSen,accSpec,accPrec,accFscore,maxAuc2,maxAcc2,minArmse2,maxBcc2,minBrmse2,Rs2,Ra2,Rb2,l1,l2,l3,SelectFeaIdx);

        end
    end
end



toc
endTime=toc;
fprintf('%fs\n',endTime);

% save T1W_DTI.mat Res_1 Res_2
% clear Res
% Res=Res_1;
% Res=Res_2;




mergeData=full(mergeData);
mergeYScore=full(mergeYScore);
