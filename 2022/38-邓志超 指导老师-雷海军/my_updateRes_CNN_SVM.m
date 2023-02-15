function Res=my_updateRes_SVR(Res,cc,Rmse,ccDec,ccTest,coord,bestCoord,l1,l2,l3,SelectFeaIdx)
% 输入 Acc,Sen,Spec,Prec,Fscore,Auc 矩阵



    meanOfAcc = mean(cc(:));
    stdOfAcc = std(cc(:));
    meanOfRmse = mean(Rmse(:));
    stdOfRmse = std(Rmse(:));
 
    if(Res.ScoreCc<meanOfAcc)
        Res.ScoreCc=meanOfAcc;  
        Res.ScoreCc_Rmse=meanOfRmse;        % 保存acc均值最大情况下，rmse的均值大小
        
        Res.ScoreCc_std=stdOfAcc;          % 保存acc均值最大情况下，acc的标准差
        
        Res.ScoreCc_dec=ccDec;             % 找到最大平均值acc下的检测得分
        Res.ScoreCc_test=ccTest;           % 找到最大平均值acc下的测试标签
        Res.ScoreCc_Coord=coord;
        Res.ScoreCc_bestCoord=bestCoord;	% cc均值最大情况下的parg parc循环最大参数下标
        Res.ScoreCc_l1=l1;
        Res.ScoreCc_l2=l2;
        Res.ScoreCc_l3=l3;
        Res.ScoreCc_SelectFeaIdx=SelectFeaIdx;
        
    end
    
    if(Res.ScoreRmse>meanOfRmse) % 均方误差最小值
        Res.ScoreRmse_Cc=meanOfAcc;         % 保存 Rmse 均值最大情况下，acc 的均值大小
        Res.ScoreRmse=meanOfRmse;
        
        Res.ScoreRmse_std=stdOfRmse;        % 保存 Rmse 均值最大情况下，Rmse的标准差
        
        Res.ScoreRmse_dec=ccDec;
        Res.ScoreRmse_test=ccTest;
        Res.ScoreRmse_Coord=coord;
        Res.ScoreRmse_bestCoord=bestCoord;  % rmse 均值最大情况下的parg parc循环最大参数下标
        Res.ScoreRmse_l1=l1;
        Res.ScoreRmse_l2=l2;
        Res.ScoreRmse_l3=l3;
        Res.ScoreRmse_SelectFeaIdx=SelectFeaIdx;
    end

    
end