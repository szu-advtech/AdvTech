function [maxCc,ccRmse,ccDec,ccTest,ccBest_X,ccBest_Y,minRmse]=...
    my_findBestCc_SVR(matCc,matRmse,matDec,matTest)

    % 找出最大cc(相关系数)以及对应的rmse、Ldec、Ltest、parg parc参数下标
    % 同时找出最小的rmse（均方误差），它没有对应的值


    minRmse=min(matRmse(:));

    maxCc=max(matCc(:)); % 按出最大准确率.可能有重复的
    seqIndex=find(matCc==maxCc); % 找出所有重复的最大值的顺序索引(按列)
    [x,y]=ind2sub(size(matCc),seqIndex); % 将索引转换成矩阵的坐标,注意 x,y是向量

    % 既然准确率最大值，存在重复，那么我们再看 accArmse 指标
    maxCc=[];
    ccRmse=[];
    ccDec=[];
    ccTest=[];
    ccBest_X=[];   % 记录最大值的X坐标
    ccBest_Y=[];   % 记录最大值的Y坐标
    
    for i=1:length(x)  % 根据最大分类准确度，根据其下标找出所有对应的其他指标
        
        ccBest_X=[ccBest_X,x(i)];
        ccBest_Y=[ccBest_Y,y(i)];

        maxCc=[maxCc,matCc(x(i),y(i))];
        ccRmse=[ccRmse,matRmse(x(i),y(i))];
        ccDec=[ccDec,matDec(x(i),y(i))];
        ccTest=[ccTest,matTest(x(i),y(i))];
    end
    
    [~,maxMapIndex]=min(ccRmse(:)); % 找出最大 accArmse.可能有重复的.如果有重复的就选择第一个，不再进一步看其他指标了
    maxCc=maxCc(maxMapIndex);
    ccRmse=ccRmse(maxMapIndex);
    ccDec=ccDec{maxMapIndex};
    ccTest=ccTest{maxMapIndex};

    ccBest_X=ccBest_X(maxMapIndex);
    ccBest_Y=ccBest_Y(maxMapIndex);
end









