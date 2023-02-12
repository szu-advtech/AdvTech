function [maxAcc,accSen,accSpec,accPrec,accAuc,accFscore,accLdec,accLtest,accTP,accFP,accBest_X,accBest_Y,maxAuc,aucLdec,aucLtest,aucTP,aucFP,aucBest_X,aucBest_Y]=...
    my_findBestAcc_SVM(matAcc,matSen,matSpec,matPrec,matAuc,matFscore,matLdec,matLtest,matTP,matFP)

    % -------------------- AUC -----------------------------
    [maxAuc,maxAucIndex]=max(matAuc(:)); % 按出最大fscores.可能有重复的
    [aucBest_X,aucBest_Y]=ind2sub(size(matAuc),maxAucIndex); % 将索引转换成矩阵的坐标,注意 x,y是向量
    aucLdec=matLdec{aucBest_X,aucBest_Y};
    aucLtest=matLtest{aucBest_X,aucBest_Y};
    aucTP=matTP{aucBest_X,aucBest_Y};
    aucFP=matFP{aucBest_X,aucBest_Y};
    
    
    % -------------------- Acc  -----------------------------
    maxAcc=max(matAcc(:)); % 按出最大fscores.可能有重复的
    seqIndex=find(matAcc==maxAcc); % 找出所有重复的最大值的顺序索引(按列)
    [x,y]=ind2sub(size(matAcc),seqIndex); % 将索引转换成矩阵的坐标,注意 x,y是向量

    % 既然fscores最大值，存在重复，那么我们再看  acc 准确率 指标
    accFscore=[];
    maxAcc=[];
    accSen=[];
    accSpec=[];
    accPrec=[];   
    accAuc=[];
    accLdec={};
    accLtest={};
    accTP={};
    accFP={};
    
    accBest_X=[];   % 记录最大值X的坐标
    accBest_Y=[];   % 记录最大值Y的坐标
    for i=1:length(x)  % 最大的fscores，根据其下标找出所有对应的其他指标
        
        
        accBest_X=[accBest_X,x(i)];
        accBest_Y=[accBest_Y,y(i)];
        
        accFscore=[accFscore,matFscore(x(i),y(i))];
        maxAcc=[maxAcc,matAcc(x(i),y(i))];        
        accSen=[accSen,matSen(x(i),y(i))];
        accSpec=[accSpec,matSpec(x(i),y(i))];
        accPrec=[accPrec,matPrec(x(i),y(i))];        
        accAuc=[accAuc,matAuc(x(i),y(i))];
        
        accLdec=[accLdec,matLdec{x(i),y(i)}];
        accLtest=[accLtest,matLtest{x(i),y(i)}];
        accTP=[accTP,matTP{x(i),y(i)}];
        accFP=[accFP,matFP{x(i),y(i)}];
    end
    
    [~,maxMapIndex]=max(accFscore(:)); % 按出最大 acc .可能有重复的.如果有重复的就选择第一个，不再进一步看其他指标了
    accFscore=accFscore(maxMapIndex);
    maxAcc=maxAcc(maxMapIndex);
    accSen=accSen(maxMapIndex);
    accSpec=accSpec(maxMapIndex);
    accPrec=accPrec(maxMapIndex);   
    accAuc=accAuc(maxMapIndex);
    
    accLdec=accLdec{maxMapIndex};
    accLtest=accLtest{maxMapIndex};
    accTP=accTP{maxMapIndex};
    accFP=accFP{maxMapIndex};
    
    accBest_X=accBest_X(maxMapIndex);
    accBest_Y=accBest_Y(maxMapIndex);

end









