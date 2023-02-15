function hyp=Hypervolume_calculation(pf,repoint)
% Hybervolume_calculation: Calculate the hypervolume of the obtained Pareto front
%% Input:
%                      Dimension                    Description
%      pf              population_size x n_obj      Obtained Pareto front     
%      repoint         1 x n_obj                    Reference point(depending on the test function)

%% Output:
%                     Description
%      hyp            Hypervolume

%Reference [Zitzler E, Thiele L, Laumanns M, et al. Performance assessment of multiobjective optimizers: an analysis and review[J].
%IEEE Transactions on Evolutionary Computation, 2003, 7(2): 117-132.]
popsize=size(pf,1);
[~,temp_index]=sort(pf(:,1));
sorted_pf=pf(temp_index,:);
pointset=[repoint;sorted_pf];
hyp=0;

for i=1:popsize
    cubei=(pointset(1,1)-pointset(i+1,1))*(pointset(i,2)-pointset(i+1,2));
    hyp=hyp+cubei;
end