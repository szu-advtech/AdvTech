classdef INDIVIDUAL  
    %此类代表一个种群，由五个矩阵组成：染色体、因素代价、因素等级、标量适应度、技能因素，横—种群大小，纵—维度/任务数
    %种群需要由initPOP初始化，评价种群个体时需要evaluate函数
    properties
        rnvec; % (genotype)--> decode to find design variables --> (phenotype) 
        factorial_costs;%因素代价
        factorial_ranks;%因素等级
        scalar_fitness;%标量适应度
        skill_factor;%技能因素
    end    
    methods        
        function object = initPOP(object,N,D,MM)            
            object.rnvec = rand(N,D);%初始化个体编码
            object.factorial_costs = inf*ones(N,MM);%初始化个体因素代价为0
            object.factorial_ranks = zeros(N,MM);%初始化个体因素等级为0
            object.scalar_fitness = zeros(N,1);%初始化个体标量适应度为0
            object.skill_factor = zeros(N,1);%初始化个体的技能因子为0
        end
        
        function [object,call] = evaluate(object,Task,pi_l,options)%适应度评价
            object.factorial_costs(:)=inf;
            call = 0;
            for i = 1:Task.M
                [object.factorial_costs(:,i),object.rnvec,calls]=CalObj(Task,object.rnvec,pi_l,options,i,object.skill_factor);
                call = call + calls;
            end
        end
    end
end