function [taskSet]=Task_G(prodata,remdata,weights,T,popNum,Vlimit)

%生成与原高维特征集相关的T个子任务
w_prodata = zeros(1,size(prodata,2));
for i = 1:size(prodata,2)
    w_prodata(i) = weights(prodata(i));
end
w_remdata = zeros(1,size(remdata,2));
for i = 1:size(remdata,2)
    w_remdata(i) = weights(remdata(i));
end
p = mean(w_prodata,'all') / (mean(w_prodata,'all') + mean(w_remdata,'all'));

task = cell(1,T);

for i = 1:T
    temp = [];
    for j = 1:size(prodata,2)
        n = rand;
        if n <= p
        temp(end+1) = prodata(j);
        end
    end
    for j = 1:size(remdata,2)
        n = rand;
        if n < 1-p
        temp(end+1) = remdata(j);
        end
    end
    task{1,i} = temp;
end

%初始化数据
taskSet{T} = {};
featureNum = size(weights,2);
for i = 1:T
    taskSet{i}.searchSpace = zeros(1,featureNum);
    temp = task{i};
    for j = 1:size(temp,2)
        taskSet{i}.searchSpace(temp(j)) = 1;
    end    
    taskSet{i}.position = rand(popNum,featureNum).*taskSet{i}.searchSpace;
    taskSet{i}.velocity  = (rand(size(taskSet{i}.position)) - 0.5) * Vlimit(2)*2;
    taskSet{i}.velocity = taskSet{i}.velocity.*taskSet{i}.searchSpace;
end

end