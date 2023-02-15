function Res=my_updateRes_SVM(Res,Acc,Sen,Spec,Prec,Auc,Fscore,Ldec,Ltest,TP,FP,coord,bestCoord,l1,l2,l3,SelectFeaIdx)
% ���� Acc,Sen,Spec,Prec,Fscore,Auc ����

Res4=my_updateRes_SVM(Res4,maxAcc4,accSen4,accSpec4,accPrec4,accAuc4,accFscore4,accLdec4,accLtest4,accTP4,accFP4,accCoord4,accBestCoord4);

    meanOfAcc = mean(Acc(:));
    meanOfSen = mean(Sen(:));
    meanOfSpec = mean(Spec(:));
    meanOfPrec= mean(Prec(:));    
    meanOfAuc = mean(Auc(:));
    meanOfFscore = mean(Fscore(:));
    
    
    stdOfAcc = std(Acc(:));
    stdOfSen = std(Sen(:));
    stdOfSpec = std(Spec(:));
    stdOfPrec = std(Prec(:));
    stdOfAuc = std(Auc(:));
    stdOfFscore = std(Fscore(:));

    
    if(Res.Acc<meanOfAcc)       % �ҵ����acc��ֵ
        Res.Acc=meanOfAcc;
        Res.Acc_std=stdOfAcc;   % ����ֵACC�µ� acc����
        Res.Acc_mat=Acc;        % ����acc��ֵ�������£�10�ν�����֤��acc������
        Res.Acc_Ldec=Ldec;      % �ҵ����ƽ��ֵacc�µļ��÷�,10�ν�����֤��Ldec������
        Res.Acc_Ltest=Ltest;
        Res.Acc_TP=TP;          % ����ֵACC�µ� TP��
        Res.Acc_FP=FP;          % ����ֵACC�µ� FP��
        Res.Acc_Coord=coord;
        Res.Acc_bestCoord=bestCoord;
        Res.Acc_SelectFeaIdx=SelectFeaIdx;
        Res.Acc_l1=l1;
        Res.Acc_l2=l2;
        Res.Acc_l3=l3;
        
    end
    
    if(Res.Sen<meanOfSen)
        Res.Sen=meanOfSen;
        Res.Sen_std=stdOfSen;
    end
    
    if(Res.Spec<meanOfSpec)
        Res.Spec=meanOfSpec;
        Res.Spec_std=stdOfSpec;
    end

    if(Res.Prec<meanOfPrec)
        Res.Prec=meanOfPrec;
        Res.Prec_std=stdOfPrec;
    end
    
    if(Res.Auc<meanOfAuc)
        Res.Auc=meanOfAuc;
        Res.Auc_std=stdOfAuc;
    end
    
    if(Res.Fscore<meanOfFscore)
        Res.Fscore=meanOfFscore;
        Res.Fscore_std=stdOfFscore;
    end
    

    
end