function [TestNNOut]=RBF_predictor(W2,B2,Centers,Spreads,TestSamIn)
% Usage: [TestNNOut]=RBF_predictor(W2,B2,Centers,Spreads,TestSamIn)
% Single RBF Predictor
% Input:
% W2             - Weights of RBF Model
% B2             - Bais of RBF Model
% Centers        - Centers of RBF Model
% Spreads        - Widths of RBF model
% TestSamIn      - Test Data

%
% Output: 
% TestNNOut      - Prediction of RBF Model for TestSamIn
%------------------------------------------------------------------------
N=size(TestSamIn,1);
TestDistance = dist(Centers',TestSamIn');
TestSpreadsMat = repmat(Spreads,1,N);
TestHiddenUnitOut = radbas(TestDistance./TestSpreadsMat);
TestNNOut = W2*TestHiddenUnitOut+B2;
TestNNOut=TestNNOut';
end