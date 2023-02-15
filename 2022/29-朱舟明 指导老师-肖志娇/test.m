%用于计算测试集的精确度和最后选择的特征数
function [featureNum,Acc] = test(trainX,trainY,testX,testY,result)
 
    feature = (result.gbest.task1.pos  .* result.gbest.task1.mask)>0.6;
    Mdl = fitcknn(trainX(:,feature),trainY,NumNeighbors=1);%模型
    pre = predict(Mdl,testX(:,feature));%预测
    featureNum = sum(feature);
    Acc = sum(pre==testY)/size(testY,1);
end

