function [objective,rnvec,calls] = CalObj(Task,rnvec,p_il,options,i,skill_factor)
% 计算种群所有个体在第i个任务的目标函数值（factorial_costs）
% Input:
% Task任务信息（维度，上下界，函数）、rnvec被评价的染色体、p_il预学习的概率、options预学习的学习器、i评价第i个任务、skill_factor被评价染色体的技能因子
% Output:
% objective被评价染色体的目标函数值、rnvec由于预学习出界，修改后的染色体、calls评价次数
    global N
    calls = 0;%评价次数初始化为0
    calind = find(skill_factor == 0 | skill_factor == i);%找到需要评价的种群个体
    objective = inf*ones(N,1);%目标函数值初始化
    d = Task.Tdims(i);
    pop = length(calind);
    objective1 = inf*ones(pop,1);%需要计算的目标函数值初始化
    nvars = rnvec(calind,1:d);
    minrange = Task.Lb(i,1:d);
    maxrange = Task.Ub(i,1:d);
    y=repmat(maxrange-minrange,[pop,1]);
    vars = y.*nvars + repmat(minrange,[pop,1]);%解码
    x = zeros(size(vars));%临时变量
    if rand(1)<=p_il
        for j = 1:pop
            [x(j,:),objective1(j),~,output] = fminunc(Task.fun(i).fnc,vars(j,:),options);%对于需要评估的个体使用拟牛顿法进行预学习
            calls=calls + output.funcCount;
        end
        nvars= (x-repmat(minrange,[pop,1]))./y;%归一化
        m_nvars=nvars;
        m_nvars(nvars<0)=0;
        m_nvars(nvars>1)=1;
        if ~isempty(m_nvars~=nvars)%如果出界
            nvars=m_nvars;
            x=y.*nvars + repmat(minrange,[pop,1]);%解码
            objective1=Task.fun(i).fnc(x);
        end
        rnvec(calind,1:d)=nvars;
    else
        x=vars;
        objective1=Task.fun(i).fnc(x);
        calls = length(calind);%适应度评价次数
    end
    objective(calind)=objective1;
end