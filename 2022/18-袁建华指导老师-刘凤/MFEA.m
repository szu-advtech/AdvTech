clc,clear
tic
%% 参数设置
global N gen
N = 50;                                     % 种群大小
rmp = 0.3;                                  % 随机交配池概率
pi_l = 1;                                   % 个体学习的概率(BFGA quasi-Newton Algorithm)
Pc = 1;                                     % 模拟二进制交叉概率
mu = 10;                                    % 模拟二进制交叉参数
sigma = 0.02;                               % 高斯变异模型的标准差
gen = 100;                                  %迭代次数
selection_process = 'elitist';              % 可供选择：elitist、roulette wheel、Tournament
name = 'RastriginAckleyLinear';             % 测试任务选择有：RastriginAckley、SphereWeierstrass、RastriginAckleySphere、RastriginRastrigin、AckleyAckley
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',5);  %调用matlab函数优化器（拟牛顿法）->设置预学习优化器


%% 初始化任务
Task = TASK();
Task = initTASK(Task,name);

%% MFEA
%%% 0.记录最优解的矩阵
EvBestFitness = zeros(gen+1,Task.M);          %每代最好的适应度值
TotalEvaluations=zeros(gen+1,1);              %每代每个个体评价次数

%%% 1.初始化种群
Population = INDIVIDUAL();                    %生成初始种群
Population = initPOP(Population,N,Task.D_multitask,Task.M);

%%% 2.根据多任务环境中的每个优化任务评估每个个体的因子代价
[Population,TotalEvaluations(1)] = evaluate(Population,Task,pi_l,options);

%%% 3.计算初始化种群的因素等级以及技能因素
[Population,EvBestFitness(1,:),bestind] = Calfactor(Population);

%%% 4.优化过程
for i = 1:gen
    %4.1 个体交叉变异
    Offspring  = GA(Population,rmp,Pc,mu,sigma);
    %4.2 计算因子代价
    [Offspring,TotalEvaluations(i+1)] = evaluate(Offspring,Task,pi_l,options);
    TotalEvaluations(i+1) = TotalEvaluations(i+1) + TotalEvaluations(i);
    %4.3 种群合并
    intpopulation = combpop(Population,Offspring);
    %4.4 更新标量适应度，技能因素，因素等级
    [intpopulation,EvBestFitness(i+1,:),bestind] = Calfactor(intpopulation);
    %4.5 环境选择
    Population = EnvironmentalSelection(intpopulation,selection_process,N,Task.M);
    disp(['MFEA Generation = ', num2str(i), ' EvBestFitness = ', num2str(EvBestFitness(i+1,:))]);%为了记录初始化的值所以次数+1
end

%% 记录算法结果
data_MFEA.wall_clock_time=toc;
data_MFEA.EvBestFitness=EvBestFitness;
data_MFEA.bestInd_data=bestind;
data_MFEA.TotalEvaluations=TotalEvaluations;
save(['Data\','data.mat'],'data_MFEA');

%% 画图
for i=1:Task.M
    figure(i)
    hold on
    plot(EvBestFitness(:,i));
    xlabel('GENERATIONS');
    ylabel(['TASK ', num2str(i), ' OBJECTIVE']);
    saveas(gcf,['Data\figure_Task',num2str(i),'.jpg']);
end