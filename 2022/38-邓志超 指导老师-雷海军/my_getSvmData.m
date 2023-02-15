

% ����ѵ�����ݡ�ѵ����ǩ���������ݡ�ѵ����ǩ��SVM����cmd
% �Խṹ�����ʽ�����ط���׼ȷ�ȡ��ٻ��ʵ�ָ��

function svmData = my_getSvmData(trainData,trainFlag,testData,testFlag,cmd)


L_SVM_MMR = svmtrain(trainFlag, trainData, cmd);

% Ldec��Ԥ���ǩ�ľ�ȷֵ��Lpre_MMR=[-1 -1 -1]��ôLdec���ܵ��ڣ�[ -0.9984 -0.9987 -0.9986 ]
% Lacc(1,1)��SVM��׼ȷ�ȣ�������SVR��׼ȷ������ Lacc(3.1)�ĸ��� ��ʾ
% [Lpre_MMR, Lacc, Ldec]
% [Lpre_MMR, Lacc, Ldec] = svmpredict(testFlag, testData, L_SVM_MMR);
[~, Lacc, Ldec] = svmpredict(testFlag, testData, L_SVM_MMR);

[FP,TP,~,lauc] = perfcurve(testFlag,Ldec,1);

% [SVMtp,SVMtn]
[SVMacc,SVMsen,SVMspec,SVMfscore,SVMprec,~,~] = my_AccSenSpe(Ldec,testFlag);

% Ldec����scores�÷֣��Ƚ������У�Ȼ�����tp fp
% ����� FP��TP �Ǹ����� �������� ���ۼӱ���

svmData.SVMacc=Lacc(1,1);   % SVM�Դ�����ķ���׼ȷ��
svmData.auc= lauc;          % area under curve (AUC) Խ��Խ��.
svmData.Ldec=Ldec;          % ����Ԥ���ǩ
svmData.Ltest=testFlag;     % ������Ա�ǩ
svmData.tp=TP;
svmData.fp=FP;

svmData.acc=SVMacc;         % SVM.acc:  accuracy ׼ȷ��
svmData.sen=SVMsen;         % SVM.sen:  sensitivity �����ԡ�����⵽����������Ŀռ����ǩ��Ŀ�ı������е�����R���ٻ��ʣ�
svmData.spec=SVMspec;       % SVM.spec: specificity �����ԡ�����⵽����������Ŀռ����ǩ��Ŀ�ı�������Ϊ -R
svmData.fscore=SVMfscore;	% SVM.Fscore: fscore f-measure: 2PR/(P+R)
svmData.prec=SVMprec;       % SVM.Prec:	precision ���� 


end

