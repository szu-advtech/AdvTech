function nPopulation = EnvironmentalSelection(Population,selection_process,N,M)
% 环境选择
% Input:Population种群，selection_process选择过程(可取elitist、roulette wheel),N选出个体的数目,M任务数
% Output:nPopulation选择之后的种群
%--------------------------------------------------------------------------
    nPopulation = INDIVIDUAL();
    if strcmp(selection_process,'elitist')%精英选择
        [~,y]=sort(Population.scalar_fitness,'descend');
    elseif strcmp(selection_process,'roulette wheel')%轮盘赌选择
        y = [];
        for i = 1:M
            ind = find(Population.skill_factor == i);%按擅长的技能进行分组选择
            Fitness = Population.scalar_fitness(Population.skill_factor == i,:);%找到分组对应的个体的适应度函数值
            ind1 = ind(RouletteWheelSelection(N/M,Fitness));%每个任务选出N/M个体，此处选出的个体索引
            y = [y;ind1];
        end
    elseif strcmp(selection_process,'Tournament')%锦标赛选择
        y = [];
        for i = 1:M
            ind = find(Population.skill_factor == i);%按擅长的技能进行分组选择
            Fitness = Population.scalar_fitness(Population.skill_factor == i,:);%找到分组对应的个体的适应度函数值
            ind1 = ind(TournamentSelection(2,N/M,Fitness));%每个任务选出N/M个体，此处选出的个体索引，默认二元锦标赛
            y = [y;ind1];
        end
    end
    nPopulation.rnvec = Population.rnvec(y(1:N),:);
    nPopulation.factorial_costs = Population.factorial_costs(y(1:N),:);
    nPopulation.factorial_ranks = Population.factorial_ranks(y(1:N),:); 
    nPopulation.scalar_fitness = Population.scalar_fitness(y(1:N),:);
    nPopulation.skill_factor = Population.skill_factor(y(1:N),:);
end