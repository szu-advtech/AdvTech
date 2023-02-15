function [F, W] = func_ALPTMR(XL, XU, YL, YU, alpha, beta, gamma)
    X = [XL, XU];
    Y = [YL, YU];
    % Initialize Q, D, O
    N = size(X, 2);
    Q = eye(N);
    D = eye(N);
    O = eye(N);
    U = diag([ones(1, size(XL, 2)), zeros(1, size(XU, 2))]);
    % Initialize Ef, Ex, Ew as O matrix
    Ef = zeros(size(Y));
    Ex = zeros(size(X));
    Ew = zeros(N);
    F = Y;
    %initialize W by LLE
    W = zeros(N,N);
    knn = 30;
    all_one_vec = ones(knn, 1);
    knn_mat = func_getKNNMat(X, knn);  %N*N 每一列中1所在行表示k近邻元素
    for i =1:N
        %求解Z
        A = [];
        for j= knn_mat(:, i)
            A = [A, X(:, i)-X(:, j)];
        end
        Z = A'*A;
        
       % Z_inv = inv(Z+0.001*eye(size(Z)));
        W_i = ((Z+0.001*eye(size(Z)))\all_one_vec)/(all_one_vec'/(Z+0.001*eye(size(Z)))*all_one_vec);    
        for j= 1:knn
            W(knn_mat(j, i), i) = W_i(j);
        end
    end

    recovered_W = W - Ew;
    recovered_X = X - Ex;
    recovered_F = F - Ef;
    %loop
    convergenced = 1;
    t = 0;
    while ~convergenced || t<30
        t = t+1;
        % update Ex
        I = eye(size(X, 2));
        A = (I - recovered_W)*(I - recovered_W)';
        Ex = X*A/(A+2*alpha*beta*Q+eye(size(A)));
        %iii = A+2*alpha*beta*Q+eye(size(A));
      
        recovered_X = X - Ex;
       
        
        F = (Ef*A-Ef*U+Y*U)/(A+U+0.01*eye(size(A)));
        
        
        Ef = (F*A-Y*U+F*U)/(A+U+2*alpha*D+0.001*eye(size(A)));

        
        recovered_F = F - Ef;
    
        H = [recovered_F; recovered_X; ones(1, size(recovered_F, 2))];

        
        
        W = (H'*H + eye(size(H, 2)))\(H'*H+Ew);
        for i = 1:size(W, 2)
            W(i, i) = 0 ;
        end
        W = max(W, 0);


        Ew = (H'*H + eye(size(H, 2)) + 2*alpha*gamma*O)\(H'*H*W+W-H'*H);
        recovered_W = W - Ew;
    
        % update Q
        for i = 1:size(Ex, 2)
            Q(i, i) = 1/((2*norm(Ex(:, i), 2)) + 1e-8);
        end
    
        % update D
        for i = 1:size(Ef, 2)
            D(i, i) = 1/((2*norm(Ef(:, i), 2)) + 1e-8);
        end
    
        % update O
        for i = 1:size(Ew, 2)
            O(i, i) = 1/((2*norm(Ew(:, i), 2)) + 1e-8);
        end
    end

end