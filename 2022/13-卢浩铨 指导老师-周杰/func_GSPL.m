function [W_hat, p_vec, z_vec] = func_GSPL(X_v, n, s, c,...
    delta, l20_k, m, Max_Iter, intra_iter)
%% introduction
% review GSPL: A Succinct Kernel Model for Group-Sparse Projections
%           Learning of Multiview Data
% Input:    X_v = multiview data input \in R^{n*d_hat}
%                   a struct array ('X_v', [], 'dv', [])
%            l20_k = the number of the non-zero rows of the final
%                   projection matrix W_hat
%            m = the dimension we want to reduce to
%            Max_Iter = for GSPL
%            max_iter = for L20-norm
%            intra_iter = for L20-norm
%            s = the number of view
%            c = clusters
%            n = number of samples
%            delta = for heated kernel function
% Output:   W_hat = [W_1, W_2, ..., W_s]
%            p_vec = the balanced parameter vector for \sum X_{v}W_{v}
%            z_vec = the balanced parameter vector for objective function

%% function body

    % construct embeddings U
    d_l = 0;
    U_total(s) = struct('U_v', []);
    Wv_mat(s) = struct('W_v', []);
    UgHXW_for_z(s) = struct('each_g', []);
    E_g_mat(s) = struct('E_g', []);
    X_hat = [];
    for i = 1: s
        X_hat = [X_hat, X_v(i).X_v];
    end

    for i = 1 : s
        d_l = d_l + X_v(i).dv;

        % get L
        S_mat = constructKernel(X_v(i).X_v, [], ...
            struct("kerneltype", "Gaussian", "t", delta));
%         S_mat = zeros(n, n);
%         for a = 1 : n
%             for b = 1 : n
%                 S_mat(a, b) = exp(-(norm(X_v(i).X_v(a, :) -...
%                     X_v(i).X_v(b, :))^2 / delta));
%             end
%         end

        D_mat = diag(sum(S_mat));
        L_mat = D_mat - S_mat;

        % get U
        [V, ~] = eigs(L_mat, c, 1e-10);
        U_total(i).U_v = V;

    end

    % construct matrix H
    H_mat = eye(n) - (1 / n) * ones(n);

    % initialize p z & W
    p_vec = ones(s, 1);
    z_vec = ones(s, 1);
    for i = 1 : s
        Wv_mat(i).W_v = eye(X_v(i).dv, m);
    end

    % initialize B D matrix
    B_mat_for20 = eye(d_l, l20_k);
    D_mat_for20 = eye(l20_k, m);
    W_hat = eye(d_l, m);

    A_mat(s) = struct('A_v', []);
    B_mat(s) = struct('B_v', []);

    for run_t = 1 : Max_Iter
        % update p and fix others
        U_hat = zeros(n, n);
        for i = 1: s
            U_hat = U_hat + z_vec(i) * H_mat' * U_total(i).U_v * (U_total(i).U_v)' * H_mat;
        end
        
        for i = 1: s
            A_mat(i).A_v = U_hat * X_v(i).X_v * Wv_mat(i).W_v;
            B_mat(i).B_v = X_v(i).X_v * Wv_mat(i).W_v;
        end
        A_hat = [];
        B_hat = [];
        for i = 1: s
            A_hat = [A_hat, reshape(A_mat(i).A_v, [], 1)];
            B_hat = [B_hat, reshape(B_mat(i).B_v, [], 1)];
        end
        [V_for_p, D_for_p] = eigs(B_hat' * A_hat);
        [~, ind] = sort(diag(D_for_p), 'descend');
        Vs_for_p = V_for_p(:, ind);
        p_vec = Vs_for_p(:, 1);
    
        % update z and fix others
        XW_for_z = zeros(size(X_v(1).X_v * Wv_mat(1).W_v));
        for i = 1: s
            XW_for_z = XW_for_z + p_vec(i) * X_v(i).X_v * Wv_mat(i).W_v;
        end
        for g = 1: s
            UgHXW_for_z(g).each_g = (U_total(g).U_v)' * H_mat * XW_for_z;
        end
        sum_of_down = 0;
        for i = 1: s
            sum_of_down = sum_of_down + (norm(UgHXW_for_z(i).each_g, 'fro'))^4;
        end
        sum_of_down = sum_of_down^(1/2);
        for g = 1: s
            z_vec(g) = ((norm(UgHXW_for_z(g).each_g, 'fro'))^2) / sum_of_down;
        end
        
        % update W and fix others
        p_hat = [];
        for i = 1: s
            p_hat = [p_hat; p_vec(i) * ones(X_v(i).dv, 1)];
        end
        P_wav = diag(p_hat);
        for i = 1: s
            E_g_mat(i).E_g = (U_total(i).U_v)' * H_mat * X_hat * P_wav;
        end
        S_E_mat = zeros(size((E_g_mat(1).E_g)' * E_g_mat(1).E_g));
        for i = 1: s
            S_E_mat = S_E_mat + z_vec(i) * (E_g_mat(i).E_g)' * E_g_mat(i).E_g;
        end

        Imm = sparse(1:m, 1:m, ones(m,1),...
            (size(S_E_mat, 1)), (size(S_E_mat, 2)));
        S_E_mat = S_E_mat + Imm;
        
        if rank(S_E_mat) <= m
            [~, Ind_se] = sort(diag(S_E_mat), 'descend');
            SE_hat = S_E_mat(Ind_se(1:l20_k), Ind_se(1:l20_k));
            for i = 1:dl-1
                for j = 1:l20_k
                    if i == Ind_se(j)
                        B_mat_for20(i, j) = 1;
                    else
                        B_mat_for20(i, j) = 0;
                    end
                end
            end
%             [~, ~, Vsvd] = svd(SE_hat);
%             D_mat_for20 = Vsvd(:, 1:m);
            [D_mat_for20, ~] = eigs(SE_hat, m, 'largestabs');
            W_hat = B_mat_for20 * D_mat_for20;
        else
            for it = 1:intra_iter
                P = S_E_mat * W_hat * pinv(W_hat' * S_E_mat * W_hat) * W_hat' * S_E_mat;
                [~, Ind_p] = sort(diag(P), 'descend');
                SE_hat = S_E_mat(Ind_p(1:l20_k), Ind_p(1:l20_k));
                for i = 1:d_l
                    for j = 1:l20_k
                        if i == Ind_p(j)
                            B_mat_for20(i, j) = 1;
                        else
                            B_mat_for20(i, j) = 0;
                        end
                    end
                end
%                 [~, ~, Vsvd] = svd(SE_hat);
%                 D_mat_for20 = Vsvd(:, 1:m);
                [D_mat_for20, ~] = eigs(SE_hat, m, 'largestabs');
                W_hat = B_mat_for20 * D_mat_for20;
            end
        end
    end
end