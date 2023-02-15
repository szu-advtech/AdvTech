classdef TASK    
    %此类为所处理的任务信息，包含任务个数，任务维度，统一搜索空间，任务上下界和任务函数，此类需要用initTASK初始化
    properties
        M;%任务个数
        Tdims;%任务维度
        D_multitask;%统一搜索空间
        Lb;%任务的下界
        Ub;%任务的上界
        fun;%函数
    end    
    methods        
        function object = initTASK(object,name)
            switch name
                case 'RastriginAckley'
                    object.M = 2;%初始化任务个数
                    object.Tdims = zeros(object.M,1);%初始化任务维度
                    object.Tdims(1) = 40;%Rastrigin维度
                    object.Tdims(2) = 30;%Ackley维度
                    object.D_multitask = max(object.Tdims);%统一搜索空间
                    object.Lb = ones(object.M,object.D_multitask);%初始化下界
                    object.Ub = ones(object.M,object.D_multitask);%初始化上界
                    object.Lb(1,:) = -5*object.Lb(1,:);%Rastrigin下界
                    object.Lb(2,:) = -32*object.Lb(2,:);%Ackley下界
                    object.Ub(1,:) = 5*object.Ub(1,:);%Rastrigin上界
                    object.Ub(2,:) = 32*object.Ub(2,:);%Ackley上界
                    MR=orth(randn(object.Tdims(1),object.Tdims(1)));%求随机矩阵的标准正交基,rotation matrices
                    object.fun(1).fnc=@(x)Rastrigin(x,MR);
                    MA=orth(randn(object.Tdims(2),object.Tdims(2)));%求随机矩阵的标准正交基,rotation matrices
                    object.fun(2).fnc=@(x)Ackley(x,MA);
                case 'SphereWeierstrass'
                    object.M = 2;%初始化任务个数
                    object.Tdims = zeros(object.M,1);%初始化任务维度
                    object.Tdims(1) = 30;%Sphere维度
                    object.Tdims(2) = 30;%Weierstrass维度
                    object.D_multitask = max(object.Tdims);%统一搜索空间
                    object.Lb = ones(object.M,object.D_multitask);%初始化下界
                    object.Ub = ones(object.M,object.D_multitask);%初始化上界
                    object.Lb(1,:) = -100*object.Lb(1,:);%Sphere下界
                    object.Lb(2,:) = -0.5*object.Lb(2,:);%Weierstrass下界
                    object.Ub(1,:) = 100*object.Ub(1,:);%Sphere上界
                    object.Ub(2,:) = 0.5*object.Ub(2,:);%Weierstrass上界
                    MS=eye(object.Tdims(1),object.Tdims(1));
                    object.fun(1).fnc=@(x)Sphere(x,MS);
                    MW=orth(randn(object.Tdims(2),object.Tdims(2)));
                    object.fun(2).fnc=@(x)Weierstrass(x,MW);
                case 'RastriginAckleySphere'
                    object.M = 3;%初始化任务个数
                    object.Tdims = zeros(object.M,1);%初始化任务维度
                    object.Tdims(1) = 40;%Rastrigin维度
                    object.Tdims(2) = 50;%Ackley维度
                    object.Tdims(3) = 20;%Sphere维度
                    object.D_multitask = max(object.Tdims);%统一搜索空间
                    object.Lb = ones(object.M,object.D_multitask);%初始化下界
                    object.Ub = ones(object.M,object.D_multitask);%初始化上界
                    object.Lb(1,:) = -5*object.Lb(1,:);%Rastrigin下界
                    object.Lb(2,:) = -32*object.Lb(2,:);%Ackley下界
                    object.Lb(3,:) = -100*object.Lb(3,:);%Sphere下界
                    object.Ub(1,:) = 5*object.Ub(1,:);%Rastrigin上界
                    object.Ub(2,:) = 32*object.Ub(2,:);%Ackley上界
                    object.Ub(3,:) = 100*object.Ub(3,:);%Sphere上界
                    MR=orth(randn(object.Tdims(1),object.Tdims(1)));
                    object.fun(1).fnc=@(x)Rastrigin(x,MR);
                    MA=orth(randn(object.Tdims(2),object.Tdims(2)));
                    object.fun(2).fnc=@(x)Ackley(x,MA);
                    MS=eye(object.Tdims(3),object.Tdims(3));
                    object.fun(3).fnc=@(x)Sphere(x,MS);
                case 'RastriginRastrigin'
                    object.M = 2;%初始化任务个数
                    object.Tdims = zeros(object.M,1);%初始化任务维度
                    object.Tdims(1) = 30;%Rastrigin维度
                    object.Tdims(2) = 30;%Rastrigin维度
                    object.D_multitask = max(object.Tdims);%统一搜索空间
                    object.Lb = ones(object.M,object.D_multitask);%初始化下界
                    object.Ub = ones(object.M,object.D_multitask);%初始化上界
                    object.Lb(1,:) = -5*object.Lb(1,:);%Rastrigin下界
                    object.Lb(2,:) = -5*object.Lb(2,:);%Rastrigin下界
                    object.Ub(1,:) = 5*object.Ub(1,:);%Rastrigin上界
                    object.Ub(2,:) = 5*object.Ub(2,:);%Rastrigin上界
                    MR=orth(randn(object.Tdims(1),object.Tdims(1)));%求随机矩阵的标准正交基,rotation matrices
                    object.fun(1).fnc=@(x)Rastrigin(x,MR);
                    object.fun(2).fnc=@(x)Rastrigin(x,MR);
                case 'AckleyAckley'
                    object.M = 2;%初始化任务个数
                    object.Tdims = zeros(object.M,1);%初始化任务维度
                    object.Tdims(1) = 30;%Ackley维度
                    object.Tdims(2) = 30;%Ackley维度
                    object.D_multitask = max(object.Tdims);%统一搜索空间
                    object.Lb = ones(object.M,object.D_multitask);%初始化下界
                    object.Ub = ones(object.M,object.D_multitask);%初始化上界
                    object.Lb(1,:) = -32*object.Lb(1,:);%Ackley下界
                    object.Lb(2,:) = -32*object.Lb(2,:);%Ackley下界
                    object.Ub(1,:) = 32*object.Ub(1,:);%Ackley上界
                    object.Ub(2,:) = 32*object.Ub(2,:);%Ackley上界
                    MA=orth(randn(object.Tdims(2),object.Tdims(2)));%求随机矩阵的标准正交基,rotation matrices
                    object.fun(1).fnc=@(x)Ackley(x,MA);
                    object.fun(2).fnc=@(x)Ackley(x,MA);
                case 'RastriginAckleyLinear'
                    object.M = 3;%初始化任务个数
                    object.Tdims = zeros(object.M,1);%初始化任务维度
                    object.Tdims(1) = 40;%Rastrigin维度
                    object.Tdims(2) = 50;%Ackley维度
                    object.Tdims(3) = 30;%Linear维度
                    object.D_multitask = max(object.Tdims);%统一搜索空间
                    object.Lb = ones(object.M,object.D_multitask);%初始化下界
                    object.Ub = ones(object.M,object.D_multitask);%初始化上界
                    object.Lb(1,:) = -5*object.Lb(1,:);%Rastrigin下界
                    object.Lb(2,:) = -32*object.Lb(2,:);%Ackley下界
                    object.Lb(3,:) = -10*object.Lb(3,:);%Linear下界
                    object.Ub(1,:) = 5*object.Ub(1,:);%Rastrigin上界
                    object.Ub(2,:) = 32*object.Ub(2,:);%Ackley上界
                    object.Ub(3,:) = 10*object.Ub(3,:);%Linear上界
                    MR=orth(randn(object.Tdims(1),object.Tdims(1)));
                    object.fun(1).fnc=@(x)Rastrigin(x,MR);
                    MA=orth(randn(object.Tdims(2),object.Tdims(2)));
                    object.fun(2).fnc=@(x)Ackley(x,MA);
                    MS=eye(object.Tdims(3),object.Tdims(3));
                    object.fun(3).fnc=@(x)Sphere(x,MS);
            end
        end  
    end
end