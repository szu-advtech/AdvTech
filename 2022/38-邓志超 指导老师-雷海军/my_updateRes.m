function Res=my_updateRes(Res,Acc,Sen,Spec,Prec,Fscore,Auc,maxAcc,minArmse,maxBcc,minBrmse,Rs,Ra,Rb,l1,l2,l3,SelectFeaIdx,W)

    %----------------------- SVM --------------------------
    meanOfAcc = mean(Acc(:));
    meanOfSen = mean(Sen(:));
    meanOfSpec = mean(Spec(:));
    meanOfPrec= mean(Prec(:)); 
    meanOfFscore = mean(Fscore(:));
    meanOfAuc = mean(Auc(:));
    
    stdOfAcc = std(Acc(:));
    stdOfSen = std(Sen(:));
    stdOfSpec = std(Spec(:));
    stdOfPrec = std(Prec(:));
    stdOfFscore = std(Fscore(:));
    stdOfAuc = std(Auc(:));

   
    % update SVM
    if Res.svmAcc < meanOfAcc           % ��������ÿһ�ε��������ҳ�A���� SVM.acc ��ֵ��󣬲������Ӧ�Ĳ���l1 l2 l3 l4(null)
        Res.svmAcc = meanOfAcc;
        Res.svmAcc_Std = stdOfAcc;
        Res.svmAcc_Fea = SelectFeaIdx;
        Res.svmAcc_W = W;
        Res.svmAcc_Rs=Rs;                % �������������ڸ�����£�10��-������㱣���100�� Rֵ��Rs Ra Rb�ֱ���SVM ��scoreA/B �ĸ�����Ϣ
        Res.svmAcc_Ra=Ra;
        Res.svmAcc_Rb=Rb;
        Res.svmAcc_Lamb=[l1,l2,l3];
        
    end
    
    if Res.svmSen < meanOfSen          
        Res.svmSen = meanOfSen;
        Res.svmSen_Std = stdOfSen;
        Res.svmSen_Lamb=[l1,l2,l3];
    end
    
    if Res.svmSpec < meanOfSpec          
        Res.svmSpec = meanOfSpec;
        Res.svmSpec_Std = stdOfSpec;
        Res.svmSpec_Lamb=[l1,l2,l3];
    end
    
    if Res.svmPrec < meanOfPrec          
        Res.svmPrec = meanOfPrec;
        Res.svmPrec_Std = stdOfPrec;
        Res.svmPrec_Lamb=[l1,l2,l3];
    end
    
    if Res.svmFscore < meanOfFscore          
        Res.svmFscore = meanOfFscore;
        Res.svmFscore_Std = stdOfFscore;
        Res.svmFscore_Lamb=[l1,l2,l3];
    end
    
    if Res.svmAuc < meanOfAuc            % auc�ǵ����ģ�����ÿ��parc pargѭ���У�ȡ�����ֵ��
        Res.svmAuc = meanOfAuc;
        Res.svmAuc_Std = stdOfAuc;
        Res.svmAuc_Lamb=[l1,l2,l3];
        Res.svmAuc_Rs=Rs;
    end
    
    
    %---------------------- scoreA ------------------------
    meanOfmaxAcc = mean(maxAcc(:));
    stdOfmaxAcc = std(maxAcc(:));  
    if Res.maxAcc < meanOfmaxAcc           
        Res.maxAcc = meanOfmaxAcc;
        Res.maxAcc_Std = stdOfmaxAcc;
        Res.maxAcc_Rs=Rs;
        Res.maxAcc_Ra=Ra;
        Res.maxAcc_Rb=Rb;
        Res.maxAcc_Lamb=[l1,l2,l3];
    end
    
    meanOfminArmse = mean(minArmse(:));
    stdOfminArmse = std(minArmse(:));
    if Res.minArmse > meanOfminArmse        % ע����������ԽСԽ��   
        Res.minArmse = meanOfminArmse;
        Res.minArmse_Std = stdOfminArmse;
        Res.minArmse_Rs=Rs;
        Res.minArmse_Ra=Ra;
        Res.minArmse_Rb=Rb;
        Res.minArmse_Lamb=[l1,l2,l3];
    end
    
    %---------------------- scoreB ------------------------
    meanOfmaxBcc = mean(maxBcc(:));
    stdOfmaxBcc = std(maxBcc(:));  
    if Res.maxBcc < meanOfmaxBcc           
        Res.maxBcc = meanOfmaxBcc;
        Res.maxBcc_Std = stdOfmaxBcc;
        Res.maxBcc_Rs=Rs;
        Res.maxBcc_Ra=Ra;
        Res.maxBcc_Rb=Rb;
        Res.maxBcc_Lamb=[l1,l2,l3];
    end
    
    meanOfminBrmse = mean(minBrmse(:));
    stdOfminBrmse = std(minBrmse(:));
    if Res.minBrmse > meanOfminBrmse        % ע����������ԽСԽ��   
        Res.minBrmse = meanOfminBrmse;
        Res.minBrmse_Std = stdOfminBrmse;
        Res.minBrmse_Rs=Rs;
        Res.minBrmse_Ra=Ra;
        Res.minBrmse_Rb=Rb;
        Res.minBrmse_Lamb=[l1,l2,l3];
    end
    
end



















