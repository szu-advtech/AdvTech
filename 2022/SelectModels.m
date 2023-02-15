function [ S ] = SelectModels(W,B,C,S,Xb,c,Q )
% Usage: [ S ] = SelectModels(W,B,C,S,Xb,c,Q )
%Selecting a subset of bagging models
% Input:
% W             - Weights of RBF Models
% B             - Bais of RBF Models
% C             - Centers of RBF Models
% S             - Widths of RBF models
% Xb            - Best Individual
% c             - Number of Decision Variables
% Q             - Number of Selected Models
%
% Output: 
% S             - Index of Selected Models
%------------------------------------------------------------------------

Y = RBF_Ensemble_predictor( W,B,C,S,Xb(:,1:c),c );
T=size(Y,2);
[A,I]=sort(Y);

S=[1:Q]';
S=ceil(S*T/Q);
S=I(S)';

end

