function [featureNum,error] = KNN_5fold(dataX,dataY,flag)
    global errorType;
    fold = 5;
    popSize = size(flag,1);
    featureNum = sum(flag,2);
    maxFeatureNum = max(featureNum);
    error = ones(popSize,1);            
    indices = crossvalind('Kfold',dataY,fold);

    for j = 1:popSize
        if featureNum(j)~=0
            err = zeros(1,fold);
            for i = 1:fold
                dataX_train = dataX(indices ~= i,:);
                dataY_train = dataY(indices ~= i,:); 
                dataX_test = dataX(indices == i,:);
                dataY_test = dataY(indices == i,:); 
                feature = flag(j,:);
                pre = KNN(dataX_train(:,feature),dataY_train,dataX_test(:,feature));
                err(i) = errorType(pre,dataY_test);
            end
            error(j) = mean(err,2);
        else
            featureNum(j) = maxFeatureNum;
        end
    end
end

