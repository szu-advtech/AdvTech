%五重交叉验证获取错误率
function [featureNum,error] = knn5foldFast(dataX,dataY,flag)
    fold = 5;
    popSize = size(flag,1);
    featureNum = sum(flag,2);
    maxFeatureNum = max(featureNum);
    error = ones(popSize,1);      
    indices = crossvalind('Kfold',dataY,fold);

    for j = 1:popSize
        if featureNum(j)~=0
            RD = 0;
            Dis = pdist2(dataX(:,flag(j,:)),dataX(:,flag(j,:)));
            Dis(logical(eye(size(Dis)))) = inf;
            for i = 1:fold
                trainXIndex = indices ~= i;
                trainYIndex = indices ~= i; 
                testXIndex = indices == i;
                testYIndex = indices == i; 
                [~, minIndex] = min(Dis(testXIndex,trainXIndex), [], 2);
                pre = dataY(trainYIndex);
                pre = pre(minIndex);%预测值
                y = dataY(testYIndex);
                tbl = tabulate(y);
                tbl(tbl(:,2)==0,:) = [];
                TPR = 0;
                for k = tbl(:,1)'%所有类别循环
                    index = k==y;
                    TPR = TPR + sum(pre(index)==y(index))/sum(index);
                end
                RD = 1-TPR/size(tbl,1) + RD;
            end
            error(j) = RD/fold;%平均错误率
        else
            featureNum(j) = maxFeatureNum;
        end
    end
end

