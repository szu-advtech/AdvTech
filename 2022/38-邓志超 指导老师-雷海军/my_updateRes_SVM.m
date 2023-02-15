function Res=my_updateRes_SVM(Res,Acc,Sen,Spec,Prec,Auc,Fscore,Ldec,Ltest,TP,FP,coord,bestCoord,l1,l2,l3,SelectFeaIdx)
% 输入 Acc,Sen,Spec,Prec,Fscore,Auc 矩阵

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

    
    if(Res.Acc<meanOfAcc)       % 找到最大acc均值
        Res.Acc=meanOfAcc;
        Res.Acc_std=stdOfAcc;   % 最大均值ACC下的 acc方差
        Res.Acc_mat=Acc;        % 保存acc均值最大情况下，10次交叉验证的acc矩阵组
        Res.Acc_Ldec=Ldec;      % 找到最大平均值acc下的检测得分,10次交叉验证的Ldec矩阵组
        Res.Acc_Ltest=Ltest;
        Res.Acc_TP=TP;          % 最大均值ACC下的 TP组
        Res.Acc_FP=FP;          % 最大均值ACC下的 FP组
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