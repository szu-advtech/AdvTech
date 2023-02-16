clear;
clc;
close all;

%读入数据
load("PCMAC.mat");
data = X;
label = Y;

%超参数设置
ReliefFParms = 10;
rmp = 0.6;
c1 = 1.49445;
c2 = 1.49445;
popNum = 50;
[~,featureNum] = size(data); 
Vlimit = [-0.6,0.6];  
alpha = 0.9;  % 平衡错误率和特征数
theta = 0.6;  % 决定特征是否选择的阈值
global knn_neighborsNum;
knn_neighborsNum  = 5;
global errorType;
errorType = @getBalanceError;
max_iter = 100;
maxIterUpdateVelocity = 6;

%把数据进行分类
[prodata,remdata,weights] = FSC(data,label);

%生成相关的T个子任务
T = 8;
[taskSet]=Task_G(prodata,remdata,weights,T,popNum,Vlimit);

%初始化种群
for task=1:T
    taskSet{task}.fitness = getFitness(data,label,taskSet{task}.position,theta,alpha);
    taskSet{task}.pBest.pos = taskSet{task}.position;
    taskSet{task}.pBest.fit = taskSet{task}.fitness;
    [~,index] = min(taskSet{task}.fitness);
    taskSet{task}.gBest.pos = taskSet{task}.position(index,:);
    taskSet{task}.gBest.fit = taskSet{task}.fitness(index,:);
    taskSet{task}.notChangeIters = 0;
end

%开始迭代
for iter = 1:max_iter
    
    disp(['*******************run for: ' num2str(iter) '****************']);
    w = 0.9-0.5*iter/max_iter;
    
    % 更新速度
    for task = 1:T
        % 这里迁移的时候对于不同搜索空间只迁移了重叠部分
        if rand()< rmp
            if taskSet{task}.notChangeIters >= maxIterUpdateVelocity
                % 这里因为只考虑重叠部分的平均，所以实现的相对繁琐一点
                    gBest = zeros(1,featureNum);
                    for i = 1:featureNum
                        if taskSet{task}.searchSpace(1,i)
                           posList = [];
                           for j = 1:T
                               if taskSet{j}.searchSpace(1,i) 
                                    posList(end+1) = taskSet{j}.gBest.pos(1,i);
                               end
                           end
                           gBest(1,i) = mean(posList);
                        end
                    end
            else
                    %这里文中锦标赛排序的参数没有交代，设置成2
                    TSNum = 2;
                    gBestList = zeros(1,T);
                    for i = 1:T
                       gBestList(i) = taskSet{i}.gBest.fit; 
                    end
                    % 防止锦标赛选择选到自己
                    index = 0;
                    while index ~= task
                        index = TournamentSelection(TSNum,1,gBestList);
                    end
                    overlapSpace = taskSet{task}.searchSpace & taskSet{index}.searchSpace;
                    gBest = taskSet{task}.gBest.pos;
                    cr = rand(1,sum(overlapSpace));
                    gBest(overlapSpace) = cr.*taskSet{task}.gBest.pos(overlapSpace) + (1-cr).*taskSet{index}.gBest.pos(overlapSpace);
            end
        else
            gBest = taskSet{task}.gBest.pos;
        end
        % 更新速度
        taskSet{task}.velocity = w*taskSet{task}.velocity + c1*rand(popNum,featureNum).*(taskSet{task}.pBest.pos-taskSet{task}.position) + ... 
                c2*rand(popNum,featureNum).*(gBest-taskSet{task}.position);
        taskSet{task}.velocity(taskSet{task}.velocity<Vlimit(1)) = Vlimit(1);
        taskSet{task}.velocity(taskSet{task}.velocity>Vlimit(2)) = Vlimit(2);
    end
    
    
    % 更新位置,pBest,gBest
    for task = 1:T
        taskSet{task}.position = taskSet{task}.position + taskSet{task}.velocity;
        taskSet{task}.position(taskSet{task}.position>1) = 1;
        taskSet{task}.position(taskSet{task}.position<0) = 0;
        % 获取fitness
        taskSet{task}.fitness = getFitness(data,label,taskSet{task}.position,theta,alpha);
        % 更新pBest
        tempIndex = (taskSet{task}.fitness<taskSet{task}.pBest.fit);
        taskSet{task}.pBest.pos(tempIndex,:) = taskSet{task}.position(tempIndex,:);
        taskSet{task}.pBest.fit(tempIndex,:) = taskSet{task}.fitness(tempIndex,:);
        % 更新gBest
        [fitValue,index] = min(taskSet{task}.fitness);
        if fitValue < taskSet{task}.gBest.fit
            taskSet{task}.gBest.fit = fitValue;
            taskSet{task}.gBest.pos =  taskSet{task}.position(index,:);
        end
    end

end

%提取各子种群的gbest和fitness
finGBest{T} = {};
for task = 1:T
    finGBest{task}.solution = taskSet{task}.gBest.pos > theta;
    finGBest{task}.Featurenum = sum(finGBest{task}.solution == 1);
    finGBest{task}.fit = taskSet{task}.gBest.fit;
end

disp('*******************end****************');
