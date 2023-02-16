function Res=my_updateRes_impro(Res,Acc,Sen,Spec,Prec,Fscore,Auc,...
    maxAcc,minArmse,maxBcc,minBrmse,maxCcc,minCrmse,maxDcc,minDrmse,Rs,Ra,Rb,Rc,Rd,...
    l1,l2,l3,SelectFeaIdx,W)



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
    if Res.svmAcc < meanOfAcc           % 三个参数每一次迭代，都找出A保存 SVM.acc 均值最大，并保存对应的参数l1 l2 l3 l4(null)
        Res.svmAcc = meanOfAcc;
        Res.svmAcc_Std = stdOfAcc;
        Res.svmAcc_Fea = SelectFeaIdx;
        Res.svmAcc_W = W;
        Res.svmAcc_Rs=Rs;                % 保存三个参数在该情况下，10次-交叉计算保存的100组 R值，Rs Ra Rb分别是SVM 和scoreA/B 的附加信息
        Res.svmAcc_Ra=Ra;
        Res.svmAcc_Rb=Rb;
        Res.svmAcc_Rc=Rc;
        Res.svmAcc_Rd=Rd;
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
    
    if Res.svmAuc < meanOfAuc            % auc是单独的，它是每次parc parg循环中，取的最大值。
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
    if Res.minArmse > meanOfminArmse        % 注意均方误差是越小越好   
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
    if Res.minBrmse > meanOfminBrmse        % 注意均方误差是越小越好   
        Res.minBrmse = meanOfminBrmse;
        Res.minBrmse_Std = stdOfminBrmse;
        Res.minBrmse_Rs=Rs;
        Res.minBrmse_Ra=Ra;
        Res.minBrmse_Rb=Rb;
        Res.minBrmse_Lamb=[l1,l2,l3];
    end
    
    %---------------------- scoreC ------------------------
    meanOfmaxCcc = mean(maxCcc(:));
    stdOfmaxCcc = std(maxCcc(:));  
    if Res.maxCcc < meanOfmaxCcc           
        Res.maxCcc = meanOfmaxCcc;
        Res.maxCcc_Std = stdOfmaxCcc;
        Res.maxCcc_Rs=Rs;
        Res.maxCcc_Ra=Ra;
        Res.maxCcc_Rb=Rb;
        Res.maxCcc_Rc=Rc;
        Res.maxCcc_Rd=Rd;
        Res.maxCcc_Lamb=[l1,l2,l3];
    end
    
    meanOfminCrmse = mean(minCrmse(:));
    stdOfminCrmse = std(minCrmse(:));
    if Res.minCrmse > meanOfminCrmse        % 注意均方误差是越小越好   
        Res.minCrmse = meanOfminCrmse;
        Res.minCrmse_Std = stdOfminCrmse;
        Res.minCrmse_Rs=Rs;
        Res.minCrmse_Ra=Ra;
        Res.minCrmse_Rb=Rb;
        Res.minCrmse_Rc=Rc;
        Res.minCrmse_Rd=Rd;
        Res.minCrmse_Lamb=[l1,l2,l3];
    end
    
    %---------------------- scoreD ------------------------
    meanOfmaxDcc = mean(maxDcc(:));
    stdOfmaxDcc = std(maxDcc(:));  
    if Res.maxDcc < meanOfmaxDcc           
        Res.maxDcc = meanOfmaxDcc;
        Res.maxDcc_Std = stdOfmaxDcc;
        Res.maxDcc_Rs=Rs;
        Res.maxDcc_Ra=Ra;
        Res.maxDcc_Rb=Rb;
        Res.maxDcc_Rc=Rc;
        Res.maxDcc_Rd=Rd;
        Res.maxDcc_Lamb=[l1,l2,l3];
    end
    
    meanOfminDrmse = mean(minDrmse(:));
    stdOfminDrmse = std(minDrmse(:));
    if Res.minDrmse > meanOfminDrmse        % 注意均方误差是越小越好   
        Res.minDrmse = meanOfminDrmse;
        Res.minDrmse_Std = stdOfminDrmse;
        Res.minDrmse_Rs=Rs;
        Res.minDrmse_Ra=Ra;
        Res.minDrmse_Rb=Rb;
        Res.minDrmse_Rc=Rc;
        Res.minDrmse_Rd=Rd;
        Res.minDrmse_Lamb=[l1,l2,l3];
    end
    
    
end



















