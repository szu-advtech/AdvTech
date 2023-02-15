function [W, b] = rmbcsik(d, sigma, p)
% Binary Codes for Shift-invariant kernels (Gaussian kernel)
%
%    Input:
%        d: the original dimension of sample.
%    sigma: the bandwidth of Gaussian kernel.
%        p: the dimension of random features
%
%   Output:
%        W: the weight matrix, Wij ~ N(0,sigma^{-2})
%        b: the bias vector    bij ~ Uniform(0,2*pi).
%
%   Usage:
%      [n, d] = size(X);     % each row of X is a sample.
%      [W, b] = rmbcsik(d);
%      Z = createRandomFourierFeatures(d, W, b, X);
%
%   Written by Junhong Zhang, 2022.11.4

d = 2 ^ ceil(log(d)/log(2));
W = zeros(d, p);

ptr = 1;
for ii = 1:floor(p/d)
    S = rand(d, 1);
    G = randn(d, 1);
    B = sign(2*rand(d, 1)-1); % rademacher random var
    
    V = fwht(diag(B)) * d;
    V = V(randperm(d), :);    % re-order Z
    V = G .* V;
    V = fwht(V) * d;
    V = S .* V / (sigma * sqrt(d));

    if p - ptr >= d
        W(:, ptr:ptr+d-1) = V;
    else
        W(:, ptr:p) = V(:, 1:p-ptr+1);
    end
    ptr = ptr + d;
end
b = rand(1,p) * 2*pi;
end