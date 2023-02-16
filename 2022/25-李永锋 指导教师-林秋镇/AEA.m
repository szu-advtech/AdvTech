classdef AEA < ALGORITHM
% <multi/many> <real> <large/none> <constrained/none>
% Large-scale multi-objective competitive swarm optimization algorithm

%------------------------------- Reference --------------------------------
% Y. Tian, X. Zheng, X. Zhang, and Y. Jin, Efficient large-scale multi-
% objective optimization based on a competitive swarm optimizer, IEEE
% Transactions on Cybernetics, 2020, 50(8): 3696-3708.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population    = Problem.Initialization();
%             Population    = EnvironmentalSelection(Population,V,(Problem.FE/Problem.maxFE)^2);
             ideal = 100000000000000000;
            nadir = -1000000000000000000;
            eNum1 = zeros(1,100);
            count = 1;
            flag = false;
            [z,znad]      = deal(min(Population.objs),max(Population.objs));
            %% Optimization
            while Algorithm.NotTerminated(Population)
                
                  while flag ==false
%                    if Problem.FE<20000
                      eNum = 0;
                   [P,LabelSet,ideal,nadir]= classification(Population,ideal,nadir);
                    t = (Problem.FE/Problem.maxFE);
%                     Lx=training(P,LabelSet,Problem,Population,t);
                   Lx = BPNN(P.decs,repmat(LabelSet.decs,size(P.decs,1),1),Population.decs);
                   
                        for i = 1:Problem.N
                            if rand <0.5
                                P1 = randperm(Problem.N,1);
                                P2 = randperm(Problem.N,1);
                                Offspring1(1,i) = OperatorDE1(Population(i),Population(P1),Population(P2),Lx(i,:),t);
                 
                            else
                                P1 = randperm(Problem.N,1);
                                P2 = randperm(Problem.N,1);
                                Offspring1(1,i) =OperatorDE(Population(i),Population(P1),Population(P2));
                            end
                        end
                  Offspring = Offspring1;
                    for i = 1: Problem.N
                    
%                      Population1 = [Population Offspring(1,i)];
                    Population1 = [Offspring Population(1,i)];
                    [FrontNo,MaxFNo] = NDSort(Population1.objs,Population1.cons,Problem.N+1);
                    if FrontNo(end)~=1
                        eNum = eNum+1;
                    end
                    end 
                    eNum1(1,count) = eNum/Problem.N;
                    
                    if eNum1(1,count) <0.15
                        flag = true;
                    end
                     [Population,FrontNo,CrowdDis] = EnvironmentalSelection1([Population,Offspring],Problem.N);   
                    count = count +1;
                    if Problem.FE >=1000000
                        flag = true;
                        
                    end
                  end
                  while size(Population,2)< 4
                      a = size(Population,2);
                      Population(1,a+1)=Population(1,randperm(size(Population,2),1));
                  end
                  Fitness = calFitness(Population.objs);
                  if length(Population) >= 2
                    Rank = randperm(length(Population),floor(length(Population)/2)*2);
                  else
                    Rank = [1,1];
                  end
                   Loser  = Rank(1:end/2);
                  Winner = Rank(end/2+1:end);
                Change = Fitness(Loser) >= Fitness(Winner);
                Temp   = Winner(Change);
                Winner(Change) = Loser(Change);
                Loser(Change)  = Temp;
                Offspring      = Operator(Population(Loser),Population(Winner));
                Population     = EnvironmentalSelection([Population,Offspring],V,(Problem.FE/Problem.maxFE)^2);
                  end
            end
        end
 end


function Fitness = calFitness(PopObj)
% Calculate the fitness by shift-based density

    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end
