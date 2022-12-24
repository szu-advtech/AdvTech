 function [iter,obj_new,F,A,b] = trpca(X,dim_k,m)
%UNTITLED 截断的鲁棒PCA
%   X: d*n  样本 
%   F: d*k  投影矩阵
%   b: d*1  最优均值
%   m: 截断的第m小的重构误差样本
%   A: n*1

[d, n] = size(X);

vecterone = ones(size(X,2),1);  % 全为1的竖向量
F = zeros(d,dim_k);
b = zeros(d,1);
A = ones(n,1);

iters = 100;
obj = zeros(1,iters);
D = eye(n);        % 初始化D为单位矩阵而不是初始化F和b
for iter = 1:iters

    % updata F
    sqrtD = sqrt(D);
    tmp0 = sqrtD * vecterone * vecterone' * D;
    tmp = vecterone' * D * vecterone;
    tmpF = (sqrtD - tmp0/tmp) * X';

    [U, S, V] = svd(tmpF);
    F = V(:,1:size(F,2));

    % update b
    b = (1/tmp) * X * D * vecterone;

    errors = Reconerror(X,F,b);    
    obj_new = errors * A;
    obj(iter) = obj_new;

    % update A
    errorsorted = sort(errors);
    truncat = errorsorted(m);


    % update A and weight D(1*n)
    for i = 1:n
        if errors(i) <= truncat
            A(i) = 1;
            D(i,i) = 1/(2*errors(i));
        else
            A(i) = 0;
            D(i,i) = 0;
        end
        
    end



    if iter>1 && obj_Old - obj_new < 0.001
        break;
    end
    obj_Old = obj_new;


end