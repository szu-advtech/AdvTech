function [fea] = func_formMyHOG(data, d0, d1)
%% Introduction
% Input     data = a matrix \in R^{d \times n}; d is the dimension of data
%               and n is the number of samples
%           d0 = row dimension of each image
%           d1 = column dimension of each image
% Output    fea = GIST feature. Actually, I dont know the output dimension.
%               But whatever, use tmd. \in R^{d' \times n}

%% Function body
    [~, n] = size(data);
    fea = [];
    for i = 1:n
        fea = [fea; extractHOGFeatures(reshape(data(:, i), [d0, d1]))];
    end
    fea = fea';
end