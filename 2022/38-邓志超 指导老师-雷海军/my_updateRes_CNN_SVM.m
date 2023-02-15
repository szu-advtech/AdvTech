function Res=my_updateRes_SVR(Res,cc,Rmse,ccDec,ccTest,coord,bestCoord,l1,l2,l3,SelectFeaIdx)
% ���� Acc,Sen,Spec,Prec,Fscore,Auc ����



    meanOfAcc = mean(cc(:));
    stdOfAcc = std(cc(:));
    meanOfRmse = mean(Rmse(:));
    stdOfRmse = std(Rmse(:));
 
    if(Res.ScoreCc<meanOfAcc)
        Res.ScoreCc=meanOfAcc;  
        Res.ScoreCc_Rmse=meanOfRmse;        % ����acc��ֵ�������£�rmse�ľ�ֵ��С
        
        Res.ScoreCc_std=stdOfAcc;          % ����acc��ֵ�������£�acc�ı�׼��
        
        Res.ScoreCc_dec=ccDec;             % �ҵ����ƽ��ֵacc�µļ��÷�
        Res.ScoreCc_test=ccTest;           % �ҵ����ƽ��ֵacc�µĲ��Ա�ǩ
        Res.ScoreCc_Coord=coord;
        Res.ScoreCc_bestCoord=bestCoord;	% cc��ֵ�������µ�parg parcѭ���������±�
        Res.ScoreCc_l1=l1;
        Res.ScoreCc_l2=l2;
        Res.ScoreCc_l3=l3;
        Res.ScoreCc_SelectFeaIdx=SelectFeaIdx;
        
    end
    
    if(Res.ScoreRmse>meanOfRmse) % ���������Сֵ
        Res.ScoreRmse_Cc=meanOfAcc;         % ���� Rmse ��ֵ�������£�acc �ľ�ֵ��С
        Res.ScoreRmse=meanOfRmse;
        
        Res.ScoreRmse_std=stdOfRmse;        % ���� Rmse ��ֵ�������£�Rmse�ı�׼��
        
        Res.ScoreRmse_dec=ccDec;
        Res.ScoreRmse_test=ccTest;
        Res.ScoreRmse_Coord=coord;
        Res.ScoreRmse_bestCoord=bestCoord;  % rmse ��ֵ�������µ�parg parcѭ���������±�
        Res.ScoreRmse_l1=l1;
        Res.ScoreRmse_l2=l2;
        Res.ScoreRmse_l3=l3;
        Res.ScoreRmse_SelectFeaIdx=SelectFeaIdx;
    end

    
end