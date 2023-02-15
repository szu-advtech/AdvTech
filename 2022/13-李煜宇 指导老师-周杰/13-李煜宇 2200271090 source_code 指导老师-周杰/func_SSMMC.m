function P = func_SSMMC(XL, XU, YL, k, sigma, lamda1, lamda2)
    %semi_supervised_MMC
    c = size(YL, 1);
    [d, nL] = size(XL);
    X = [XL, XU];
    n_total = size(X, 2);
    % Calculate Sb and Sw
    %mean_mat = [];
    Sw = zeros(d, d);
    Sb = zeros(d, d);
    total_mean = mean(XL, 2);
    for c_i = 1:c
        X_ci = [];
        for j = 1:nL
            if YL(c_i, j) == 1
                X_ci = [X_ci, XL(:, j)];
            end            
        end
        mean_vec = mean(X_ci, 2);
        %mean_mat = [mean_mat, mean_vec];
        Sw_i = X_ci - mean_vec*ones(1, size(X_ci, 2));
        Sw_i = Sw_i*Sw_i';
        Sw = Sw + Sw_i;
        Sb = Sb+ size(X_ci, 2)*(mean_vec-total_mean)*(mean_vec-total_mean)';
    end

    % Calculate XLX'
    M = zeros(n_total, n_total);
    I = eye(n_total, n_total);

    knn_mat = func_getKNNMat(X, 30);
    for i = 1:n_total
        for j = knn_mat(:, i)
            M(i, j) = exp(-norm(X(:,i)-X(:,j))^2/(2*sigma^2));
        end
    end
    D = diag(sum(M, 2));
    L = I - (D^(-1/2))*M*(D^(-1/2));

    %calculate P
    var = Sb - lamda1*Sw - lamda2*X*L*X';
    [P, D_value]=eig(var);
    [~, ind] = sort(diag(D_value), 'descend');
    P = P(:, ind);
    P = P(:, 1:k);
end
