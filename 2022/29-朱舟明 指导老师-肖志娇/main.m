clear;
clc;

dataList = {'DLBCL','9Tumor','Leukemia1'};
% dataList = {'9Tumor'};
for name = 1:numel(dataList)
    tic;%开启秒表计时器
    dataName = dataList{name};
    load(['data/' dataName]);
    dataX = mapminmax(X',0,1)';%归一化
    dataY = Y;
    clear X Y;
    foldNum = 10;
    [newIDX,promisingFeature.KN_point,promisingFeature.weights] = featureSelect(dataX,dataY,dataName);
    promisingFeature.subset = false(1,size(dataX,2));
    promisingFeature.subset(1:promisingFeature.KN_point) = true;
    dataX = dataX(:,newIDX);
    promisingFeature.weights = promisingFeature.weights(newIDX);

    result = {};


    repeat=30;

    for i = 1:repeat
        indices = crossvalind('Kfold',dataY,foldNum);
        depen = {};
        for fold = 1:foldNum %十重交叉验证
            boo=indices==fold;
            testX = dataX(indices == fold,:);
            testY = dataY(indices == fold,:); 
            trainX = dataX(indices ~= fold,:);
            trainY = dataY(indices ~= fold,:); 

            res = PSO_EMT(trainX, trainY, dataName,fold,promisingFeature);
            [res.featureNum,res.acc] = test(trainX,trainY,testX,testY,res);

            res2.acc=res.acc;
            res2.featureNum=res.featureNum;
            depen{fold}=res2;
            disp(strcat("PSO_EMT on, ", dataName,'run == ',num2str(i)," fold ==", num2str(fold),' ,feature num ==',num2str(res.featureNum),' ,test Acc ==',num2str(res.acc),' ,time==',num2str(toc)));
        end
        result{i} = depen;

    end

    total_time=toc;
    save(['result/' dataName],'result','foldNum','repeat','total_time');

end
