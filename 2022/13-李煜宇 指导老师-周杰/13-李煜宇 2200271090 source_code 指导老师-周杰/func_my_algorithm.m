function P = func_my_algorithm(XL, XU, YL, max_iter, k, alpha, beta, gamma, epsilon)
    %initilize P
    P_RR = RR(XL, YL);
    P = P_RR;
    X = [XL, XU];
    N = size(X, 2);
    c = size(YL, 1);
    for iter = 1:max_iter
        %update Y
        Y0 = P'*XU;
        YU_predict = zeros(size(Y0));
        [~, c_index] = max(Y0);
        for i =1:size(XU, 2)
            YU_predict(c_index(i), i) = 1;
        end
        Y  = [YL, YU_predict];

        %update Dr
        Dr = zeros(N, N);
        for i = 1:N
            Dr(i, i) = 1/(norm(Y(:, i)-P'*X(:, i) ,2)+(1e-8));
        end
    
        %update XD
        X_c_means = [];
        XD = [];
        for i = 1:c         
            X_i = [];  
            for j =1:N
                if Y(i, j)==1
                    X_i = [X_i, X(:, j)];
                end
            end            
            X_c_means = [X_c_means, (sum(X_i, 2)/size(X_i, 2))*(ones(1, size(X_i, 2)))];
            XD = [XD, X_i];
        end
        XD = XD - X_c_means;

        %update DD
        DD = zeros(N, N);
        FD = P'*XD;
        for i = 1:N
            t = norm(FD(:, i), 2);
            if t <=epsilon
                DD(i, i) = 1/(t+(1e-8));
            end
        end


        %update W by LLE        
        W = zeros(N,N);
        F = P'*X;
        all_one_vec = ones(k, 1);
        knn_mat = func_getKNNMat(F, k);  %N*N 每一列中1所在行表示k近邻元素
        for i =1:N
            %求解Z
            A = [];
            for j= knn_mat(:, i)
                A = [A, F(:, i)-F(:, j)];
            end
            Z = A'*A;
            
            Z_inv = inv(Z+0.001*eye(size(Z)));
            W_i = (Z_inv*all_one_vec)/(all_one_vec'*Z_inv*all_one_vec);    
            for j= 1:k
                W(knn_mat(j, i), i) = W_i(j);
            end
        end

        %update Dp
        row = size(P, 1);
        Dp = zeros(row, row);
        for i=1:row
            Dp(i, i) = 1/(norm(P(i, :), 2)+(1e-8));
        end
        %update P
        I = eye(size(W));
        var = (1-alpha)*(I-W)*(I-W)'+alpha*Dr;  
        %{
        a = size(X)
        b = size(XD)
        c=size(Dr)
        d=size(Dp)
        e=size(Y)
        %}
        P = alpha*(X*var*X'+beta*XD*DD*XD'+gamma*Dp+0.001*eye(size(Dp)))\X*Dr*Y';
    end
end