function [Y] = func_dpdateY(P, X)
%FUNC_DPDATEY 此处显示有关此函数的摘要
%   此处显示详细说明
    Y = P'*X;

    new_Y = zeros(size(Y));
    [~, c_index] = max(Y);
    for i =1:size(X, 2)
        new_Y(c_index(i), i) = 1;
    end
    Y = new_Y;
end

