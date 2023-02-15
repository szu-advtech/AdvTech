function [F] = func_PNLP(XL, XU, YL, YU, sigma, miu, miu1, miu2)
%PNLP 此处显示有关此函数的摘要
%   此处显示详细说明
    X = [XL, XU];
    n_total = size(X, 2);
    % Calculate XLX'
    W = zeros(n_total, n_total);   
    k=10;
    knn_mat = func_getKNNMat(X,k);
    for i = 1:n_total
        for j = 1:k
            W(i, knn_mat(j, i)) = exp(-norm(X(:,i)-X(:,knn_mat(j, i)))^2/(sigma^2));
        end
    end
    D = diag(sum(W, 2));
    I = eye(n_total, n_total);
    L = (D^(-1/2))*W*(D^(-1/2));

    % Calculate positive labels and negative labels
    Y_positive = [YL, YU];
    Y_negative = ~Y_positive;


    % Calculate the optimal F
    I = eye(size(L));
    F = miu*(L+miu*(miu1-miu2)*I)\(miu1*Y_positive-miu2*Y_negative)';
end

