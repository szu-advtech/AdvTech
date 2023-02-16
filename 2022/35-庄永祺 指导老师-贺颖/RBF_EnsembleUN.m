function [ W,B,C,S] = RBF_EnsembleUN( offline_data,D,nc,T,upperbound)
% Usage: [ W,B,C,S] = RBF_EnsembleUN( offline_data,D,nc,T,upperbound)
%Build RBF Model Pool
% Input:
% offline_data             - Offline Data with c Decision Variables and Exact Objective Value
% D             - Number of Decision Variables
% nc            - Number of neurons of RBF models
% T             - Number of RBF models
% upperbound    - The upperbound of search space
%
% Output: 
% W             - Weights of RBF Models
% B             - Bais of RBF Models
% C             - Centers of RBF Models
% S             - Widths of RBF models
%
%------------------------------- Copyright --------------------------------
% Copyright 2020. You are free to use this code for research purposes.All 
% publications which use this code should reference the following papaer:
% Jian-Yu Li, Zhi-Hui Zhan, Hua Wang, Jun Zhang, Data-Driven Evolutionary 
% Algorithm With Perturbation-Based Ensemble Surrogates, IEEE Transactions 
% on Cybernetics, DOI: 10.1109/tcyb.2020.3008280.
%--------------------------------------------------------------------------
W=zeros(T,nc);
B=zeros(1,T);
C=zeros(D,nc,T);
S=zeros(nc,T);


[row,~]=size(offline_data);
traindata=offline_data;
for i=1:T

    [ W2,B2,Centers,Spreads ] = RBF( traindata(:,1:D),traindata(:,D+1),nc);
    
    W(i,:)=W2;
    B(i)=B2;
    C(:,:,i)=Centers;
    S(:,i)=Spreads;
    
    prey=RBF_Ensemble_predictor( W(i,:),B(i),C(:,:,i),S(:,i), offline_data(:,1:D),D );
    d=prey-( offline_data(:,D+1));
    
    addid=find(d>median(d));
    rmat=upperbound*2*unifrnd(0,0.000001,row,D+1)/sqrt(D);
    rmat(:,D+1)=rmat(:,D+1)*0;%D+1 dimension is the fitness
    rl=offline_data+rmat;
    
    traindata=[ offline_data;rl(addid,:)];
end

end

