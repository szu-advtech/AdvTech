function [ Y ] = RBF_Ensemble_predictor( Weight,Bias,Centers,Spreads,Decision_vairables,D )
% Usage: [ Y ] = RBF_Ensemble_predictor( W,B,C,S,U,c )
%RBF Predictors 
% Input:
% W             - Weights of RBF Models
% B             - Bias of RBF Models
% C             - Centers of RBF Models
% S             - Widths of RBF models
% U             - Test Data with c Decision Variables
% c             - Number of Decision Variables
%
% Output: 
% Y             - Predictions of RBF Models for U
%------------------------------------------------------------------------
Y=[];
T=size(Bias,2);%Number of RBF models
for i=1:T
    Y_i=RBF_predictor(Weight(i,:),Bias(i),Centers(:,:,i),Spreads(:,i),Decision_vairables(:,1:D));
    Y=[Y,Y_i];
end

end

