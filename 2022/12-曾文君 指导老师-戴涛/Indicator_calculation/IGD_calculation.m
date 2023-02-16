function  IGD=IGD_calculation(obtained_ps,reference_ps)
% IGD_calculation: Calculate the IGD of the obtained Pareto set
% Dimension: n_var --- dimensions of decision space

%% Input:
%                      Dimension                                                   Description
%      obtained_ps     population_size x n_var                                     Obtained Pareto set     
%      reference_ps    num_of_solutions_in_reference_ps x n_var                    Reference Pareto set

%% Output:
%                     Description
%      IGD            Inverted Generational Distance (IGD) of the obtained Pareto set

n_ref=size(reference_ps,1);

for i=1:n_ref
    ref_m=repmat(reference_ps(i,:),size(obtained_ps,1),1); 
    d=obtained_ps-ref_m;    %Calculate the the differences btween the obtained_ps and the reference_ps
    D=sum(abs(d).^2,2).^0.5;%Calculate the the distance btween the obtained_ps and the reference_ps
    obtained_to_ref(i)=min(D);
end
IGD=sum(obtained_to_ref)/n_ref;
end