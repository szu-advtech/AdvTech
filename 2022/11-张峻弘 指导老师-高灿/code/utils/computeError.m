function [error, fp, fn] = computeError(Z, w, meany, y)
% COMPUTEERROR computes classification errors for ridge regression models
% that are trained using zero-mean label vectors
% Inputs:
% Z the random feature matrix
% w the optimized classication parameter vector
% meany the mean of ytrain used to train w
% y the true labels for datapoints in Z

pred = sign(Z'*w+meany);
error = mean(pred~=y);
fn = mean(pred~=y & y==1)/error;
fp = mean(pred~=y & y==-1)/error;

end