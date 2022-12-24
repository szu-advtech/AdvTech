function errors = Reconerror(X,F,b)
%UNTITLED 重构误差
%   X: d*n  样本 
%   F: d*k  投影矩阵
%   b: d*1  最优均值
%   error: 1*n 重构误差

errors = zeros(1, size(X,2));
identityM = eye(size(F,1));   % 单位矩阵
tmp = (identityM - F*F')*(X - b*ones(1,size(X,2)));
errors = sqrt(sum(tmp.^2));
end

