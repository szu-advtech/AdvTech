function [ intpopulation ] = combpop( Population,Offspring )
%合并父代和子代种群
% input:父代种群和子代种群
% output:输出后的种群
intpopulation = INDIVIDUAL();
intpopulation.rnvec = [Population.rnvec;Offspring.rnvec];
intpopulation.factorial_costs = [Population.factorial_costs;Offspring.factorial_costs];
intpopulation.factorial_ranks = [Population.factorial_ranks;Offspring.factorial_ranks];
intpopulation.skill_factor = [Population.skill_factor;Offspring.skill_factor];
intpopulation.scalar_fitness = [Population.scalar_fitness;Offspring.scalar_fitness];
end

