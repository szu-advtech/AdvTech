%获取个体对任务的适应度
function [fitness,newSkillFactor] = getFitness(dataX,dataY,position,subset,skillFactor)

    popSize = size(position,1);
    newSkillFactor = skillFactor(randi(popSize,popSize,1));
    alpha = [0.999999 0.9];
    featureNum = zeros(popSize,1);
    error = zeros(popSize,1);
    fitness = zeros(popSize,1);
    task1 = newSkillFactor==1;
    task2 = newSkillFactor==2;
    
    [featureNum(task1),error(task1)] = knn5foldFast(dataX,dataY,(position(task1,:).*subset)>0.6);
    [featureNum(task2),error(task2)] = knn5foldFast(dataX,dataY,position(task2,:)>0.6);
    
    fitness(task1) = alpha(1) * error(task1) + (1-alpha(1))*(featureNum(task1)./sum(subset));
    fitness(task2) = alpha(2) * error(task2) + (1-alpha(2))*(featureNum(task2)./size(dataX,2));

end

