function [objective,a,b] = Calfactor(objective)
% 计算标量适应度、技能因素、因素等级
% 输入：population种群的各种信息
% 输出：scalar_fitness标量适应度、skill_factor技能因素、factorial_ranks因素等级、a每个任务中最好的值、%最好的因素代价对应的个体染色体信息
    [N,~] = size(objective.factorial_costs);
    %计算因素等级
    [x,y] = sort(objective.factorial_costs);%按照因素代价的值进行升序排序
    [~,objective.factorial_ranks] = sort(y);%排序后的顺序即为其因素等级
    %Population.factorial_ranks(y) = repmat([1:N]',[1,Task.M]);%排序后的顺序即为其因素等级
    a = x(1,:);%最好的因素代价
    bb = y(1,:)';%最好的因素代价对应的个体索引
    b = objective.rnvec(bb,:);%最好的因素代价对应的个体染色体信息
    
    %计算技能因素
    [xx,yy] = min(objective.factorial_ranks,[],2);%yy为个体擅长的任务编号，xx为个体擅长的任务等级
    for i=1:N
        x = find(objective.factorial_ranks(i,:) == xx(i));%第i个染色体的表现最好的任务
        equivalent_skills = length(x);%第i个染色体的表现最好的任务数
        if(equivalent_skills == 1)
            objective.skill_factor(i) = yy(i);
        else
            objective.skill_factor(i) = x(1+round((equivalent_skills-1)*rand(1)));
        end
        %计算标量适应度
        objective.scalar_fitness(i) = 1./objective.factorial_costs(i,objective.skill_factor(i));
    end
    %计算标量适应度
        objective.scalar_fitness = 1./min(objective.factorial_ranks,[],2);
end