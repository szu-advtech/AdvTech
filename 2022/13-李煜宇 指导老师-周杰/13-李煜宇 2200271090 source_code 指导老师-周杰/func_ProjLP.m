function [P, F] = func_ProjLP(XL, XU, YL, YU, k, alpha)
    X = [XL, XU];
    Y = [YL, YU];
    n_total = size(X, 2);
    l = size(XL, 2);
    u = size(XU, 2);
    c = size(Y, 1);
    % Initialization  
    YL = [YL; zeros(1, l)];
    YU = [YU; ones(1, u)];
    Y = [YL, YU];
    P = eye(c+1);
    F = Y;
    Dp = eye(c+1);
    % Construct the neighborhood graph and define the normalized weights;
    % The regularization parameter μ i is set to μ l = 10 10 for the labeled data x i , and is set to α u= 10 −10 for unlabeled data x i in our simulations.
    W = zeros(n_total, n_total);   
    knn_mat = func_getKNNMat(X,k);
    all_one_vec = ones(k, 1);
    for i =1:n_total
        %求解Z
        A = [];
        for j= knn_mat(:, i)
            A = [A, X(:, i)-X(:, j)];
        end
        Z = A'*A; 
        size(Z);
        size(eye(k));
        Z_inv = inv(Z+0.001*eye(k, k));
        W_i = (Z_inv*all_one_vec)/(all_one_vec'*Z_inv*all_one_vec);    
        for j= 1:k
            W(knn_mat(j, i), i) = W_i(j);
        end
    end
    W = (W + W')/2;
    D = diag(sum(W, 2));
    %I = eye(n_total, n_total);
    W = (D^(-1/2))*W*(D^(-1/2));
    V = diag(sum(W, 2));
    U = zeros(n_total);
    for i = 1:n_total
        if i<=l
            U(i, i) = 10e10;
        else
            U(i, i) = 10e-10;
        end
    end

    

    I = eye(n_total);
    Q = (I - W)'*(I - W);

    convergenced = 0;
    while ~convergenced
        flag0 = P'*F;
        % Fix P and update the “shallow” soft labels F t
        F = pinv(P*P')*P*Y*U*V/(Q+U*V);
        % Fix F and Θ , and update P t+1 by
        P = (F*Q*F' + F*U*V*F' + alpha*Dp)\F*U*V*Y';
        % Fix P and update the auxiliary matrix Θ t+1
        for i = 1:c+1
            Dp(i, i) = 1/norm(P(i, :), 2);
        end
        
        flag1 = P'*F;
        err = norm(flag1-flag0, "fro");
        if err <= 1e-6
            convergenced = 1;
        end
    end

end

