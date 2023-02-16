%初始化种群
function [position,maxValue] = initPop(popSize,featureNum,weights,upperBound,KN_point)

    adjustNum = sum(weights>0)-KN_point;
    W0_num=sum(weights>0);
    middle=1-(([KN_point:W0_num]-KN_point)* (1-upperBound)/(adjustNum+1) );
    maxValue=[ones(1,KN_point-1),middle,upperBound*ones(1,featureNum-W0_num)];
    
    maxValue = repmat(maxValue,100,1);
    position = rand(popSize,featureNum);%每个样本的每一个特征值都赋值了一个0到1之间的随机值
    position = position.*maxValue;%缩小搜索空间
end

