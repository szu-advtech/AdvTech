

% 输入训练数据、训练标签、测试数据、训练标签、SVM参数cmd
% 以结构体的形式：返回分类准确度、召回率等指标

function svmData = my_getSvmData(trainData,trainFlag,testData,testFlag,cmd)


L_SVM_MMR = svmtrain(trainFlag, trainData, cmd);

% Ldec是预测标签的精确值，Lpre_MMR=[-1 -1 -1]那么Ldec可能等于：[ -0.9984 -0.9987 -0.9986 ]
% Lacc(1,1)是SVM的准确度，上面是SVR的准确度是由 Lacc(3.1)的根号 表示
% [Lpre_MMR, Lacc, Ldec]
% [Lpre_MMR, Lacc, Ldec] = svmpredict(testFlag, testData, L_SVM_MMR);
[~, Lacc, Ldec] = svmpredict(testFlag, testData, L_SVM_MMR);

[FP,TP,~,lauc] = perfcurve(testFlag,Ldec,1);

% [SVMtp,SVMtn]
[SVMacc,SVMsen,SVMspec,SVMfscore,SVMprec,~,~] = my_AccSenSpe(Ldec,testFlag);

% Ldec就是scores得分，先降序排列，然后计算tp fp
% 这里的 FP和TP 是负样本 和正样本 的累加比例

svmData.SVMacc=Lacc(1,1);   % SVM自带计算的分类准确度
svmData.auc= lauc;          % area under curve (AUC) 越大越好.
svmData.Ldec=Ldec;          % 保存预测标签
svmData.Ltest=testFlag;     % 保存测试标签
svmData.tp=TP;
svmData.fp=FP;

svmData.acc=SVMacc;         % SVM.acc:  accuracy 准确度
svmData.sen=SVMsen;         % SVM.sen:  sensitivity 敏感性——检测到正样本的数目占正标签数目的比例，有点类似R（召回率）
svmData.spec=SVMspec;       % SVM.spec: specificity 特异性——检测到负样本的数目占负标签数目的比例，视为 -R
svmData.fscore=SVMfscore;	% SVM.Fscore: fscore f-measure: 2PR/(P+R)
svmData.prec=SVMprec;       % SVM.Prec:	precision 精度 


end

