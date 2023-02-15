% RunFunction1(groupSelect,mergeDataSlect,kFold,subFold,svmParc,svmParg,parsLambda1,parsLambda2,parsLambda3,matName);

% parsIte=50; % 迭代次数
% meanSelect=0;
% RunFunction2(groupSelect,mergeDataSlect,kFold,subFold,svmParc,svmParg,parsLambda1,parsLambda2,parsLambda3,parsIte,meanSelect,matName);

% groupSelect={'isNC','isPD','NC','PD'};           % 选择不同组进行实验 NCvsPD NCvsSWEDD PDvsSWEDD
% groupSelect={'isNC','isSWEDD','NC','SWEDD'};
% groupSelect={'isPD','isSWEDD','PD','SWEDD'};

% mergeDataSlect={'X1','X2','Y','Z','M'}; % X1：灰质   X2：白质   %Y：DTI    %Z生物CSF    %M:T1CSF


clear

meanSelect=0.1;

% addpath(genpath('d:\我的文档\MATLAB\Parkinson\libsvm-mat-2.91-1'));

addpath(genpath('..\'));
% rmpath(genpath('..\'));

 % 得到正在运行的当前.m文件的名称， 通过m文件的名称，程序自动选择实验组和特征组。m文件名称的格式：*_num1num2. eg:R_11
mName=mfilename;            % 例如文件名称为：R_11，那么程序会选择1+1的组合  也就是 NCvsPD T1G
index=strfind(mName,'_');
num1=mName(index(1)+1);     % 只看第一个下划线'_'后面的两个数字
num2=mName(index(1)+2);


combineSelect1=str2double(num1); % 选择三种分组组合之一   NCvsPD NCvsSWEDD PDvsSWEDD
combineSelect2=str2double(num2); % 选择八种特征组合之一

 % 选择不同组进行实验 NCvsPD NCvsSWEDD PDvsSWEDD
groupSelectCell{1}={'isNC','isPD','NC','PD'};          
groupSelectCell{2}={'isNC','isSWEDD','NC','SWEDD'};
groupSelectCell{3}={'isPD','isSWEDD','PD','SWEDD'};

% X1：灰质   % X2：白质   % Y：DTI    % Z生物CSF    % M:T1CSF
mergeDataSelectCell{1}={'X1'};              % T1G  
mergeDataSelectCell{2}={'M'};               % T1C
mergeDataSelectCell{3}={'Y'};               % DTI
mergeDataSelectCell{4}={'X1','M'};          % T1G + T1C
mergeDataSelectCell{5}={'X1','Y'};          % T1G + DTI
mergeDataSelectCell{6}={'Y','M'};           % T1C + DTI
mergeDataSelectCell{7}={'X1','Y','M'};      % T1G + T1C + DTI
mergeDataSelectCell{8}={'X1','Y','Z','M'};  % T1G + T1C + DTI + CSF


groupSelect=groupSelectCell{combineSelect1};
mergeDataSelect=mergeDataSelectCell{combineSelect2};


kFold=2;    % 对样本进行 kFold 次划分
subFold=2; % 每次划分的子集个数


svmParc=-5:1:5; % libsvm的参数设置：2的-5次方—>2的5次方
svmParg=-5:1:5;

parsLambda1=0;
parsLambda2=0;
parsLambda3=10;

% t1=-5:2; parsLambda1 = 10.^t1;
% t2=-5:2; parsLambda2 = 10.^t2;
% t3=2:8;  parsLambda3 = 10.^t3;


% 保存的mat名字
matNameCell1={{'NCvsPD'},{'NCvsSW'},{'PDvsSW'}};
matNameCell2={{'T1G'},{'T1C'},{'DTI'},{'T1G_T1C'},{'T1G_DTI'},{'T1C_DTI'},{'T1G_T1C_DTI'},{'T1G_T1C_DTI_CSF'}}; 
matName=[num2str(combineSelect1),'.',num2str(combineSelect2),'.Mat_',...
    matNameCell1{combineSelect1}{1},'_',matNameCell2{combineSelect2}{1},'_M3T_mean',num2str(meanSelect*100),'.mat'];


parsIte=50; % 迭代次数

% RunFunction1(groupSelect,mergeDataSelect,kFold,subFold,svmParc,svmParg,parsLambda1,parsLambda2,parsLambda3,matName);
RunFunction2(groupSelect,mergeDataSelect,kFold,subFold,svmParc,svmParg,parsLambda1,parsLambda2,parsLambda3,parsIte,meanSelect,matName);


