function [fitness] = getFitness(dataX,dataY,position,theta,alpha)
    [featureNum,error] = KNN_5fold(dataX,dataY,position>theta);
    fitness = alpha*error + (1-alpha)*featureNum/size(dataX,2);
end

