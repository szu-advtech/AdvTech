function [time,P,gbest ] =DDEA_PES(D,offline_data,upperbound,lowerbound)
% Usage: [time,gbestP]=DDEA_PES(D,offline_data,upperbound,lowerbound)
%
% Input:
% offline_data  - Offline Data with c Decision Variables and Exact Objective Value
% D             - Number of Decision Variables
% upperbound    - Upper Boundary of D Decision Variables
% lowerbound    - Lower Boundary of D Decision Variables
%
% Output: 
% time          - Optimization Time
% gbest         - Final Predicted Optimum with D Decision Variables
%
%------------------------------- Copyright --------------------------------
% Copyright 2020. You are free to use this code for research purposes.All 
% publications which use this code should reference the following papaer:
% Jian-Yu Li, Zhi-Hui Zhan, Hua Wang, Jun Zhang, Data-Driven Evolutionary 
% Algorithm With Perturbation-Based Ensemble Surrogates, IEEE Transactions 
% on Cybernetics, DOI: 10.1109/tcyb.2020.3008280.
%--------------------------------------------------------------------------
rand('state',sum(100*clock));
nc=D;%Number of neurons of RBF models
fold=100;
T=fold;%Number of RBF models
%Build Model Pool
[ W,B,C,S] = RBF_EnsembleUN(offline_data,D,nc,T,upperbound(1));
%-------------------------------------------
gmax=500;
pc=1;%Crossover Probability
pm=1/D;%Mutation Probability
n=50;%Population Size
%Online Optimization-------------------------------------------
tic;
% POP = initialize_pop(n,D,upperbound,lowerbound);
%RBF Predictors 
% Y= RBF_Ensemble_predictor( W(1,:),B(1),C(:,:,1),S(:,1),POP,D );
% POP=[POP,Y];
g=1;
gbest=[];
I=(1);
POP=offline_data;
while g<=gmax
    %Model Management   
    if g~=1
        
        ri=ones(1,fold);
        I=find(ri>0.95);
        
        POP=POP(:,1:D);
        Y= RBF_Ensemble_predictor(W(I,:),B(I),C(:,:,I),S(:,I),POP,D );
        POP=[POP,Y];
    end
    %Variations    
    NPOP1=SBX(POP,upperbound,lowerbound,pc,n );
    [ Y ] = RBF_Ensemble_predictor( W(I,:),B(I),C(:,:,I),S(:,I),NPOP1,D );
    NPOP1=[NPOP1,Y];
    NPOP2=mutation(POP,upperbound,lowerbound,pm,n);
    [ Y ] = RBF_Ensemble_predictor( W(I,:),B(I),C(:,:,I),S(:,I),NPOP2,D );
    NPOP2=[NPOP2,Y];
    POP=[POP;NPOP1;NPOP2];
    %Model Combination
    YAVE=mean(POP(:,D+1:end),2);
    [A,Is]=sort(YAVE);
    POP=[POP(Is(1:n),1:D)];
    g=g+1;
    P= POP(1,1:D);
    gbest=[gbest;P];

end

toc;
time=toc;

end

