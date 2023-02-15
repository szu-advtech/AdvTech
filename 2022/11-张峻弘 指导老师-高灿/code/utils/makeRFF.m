function Z = makeRFF(W, b, X) 
%makeRFF creates Gaussian random features
% Inputs:
% D the number of features to make
% W, b the parameters for those features (d x D and 1 x D)
% X the datapoints to use to generate those features (d x N)
    D = size(W, 2);
    Z = sqrt(2/D)*cos(bsxfun(@plus,W'*X, b'));
end