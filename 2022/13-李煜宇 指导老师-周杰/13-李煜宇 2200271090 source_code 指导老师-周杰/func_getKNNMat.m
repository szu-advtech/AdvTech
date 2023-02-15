function [knn_mat] = func_getKNNMat(data,k)   %out_mat
%% 获得数据的K近邻矩阵
% Input：data，维度d*n
%        k,取k个近邻点
% Output：knn_mat，维度k*n，其值代表第i个样本的k近邻的数据下标

%% function body
    [~,n] = size(data);
    knn_mat = zeros(k,n);
    out_mat = zeros(n,n);
    for i = 1:n
        norm_k = zeros(1,k);
        index_k = zeros(1,k);
        for j = 1:k
            norm_k(j) = Inf;
        end
        for j = 1:n
            if (i~=j)
                eij = norm(data(:,i)-data(:,j));
                if eij < norm_k(k)
                    norm_k(k) = eij;
                    index_k(k) = j;
                end
                [~,sort_ind] = sort(norm_k);
                sort_norm_k = norm_k(sort_ind);
                sort_index_k = index_k(sort_ind);
                norm_k = sort_norm_k;
                index_k = sort_index_k;
            end
        end
        knn_mat(:,i) = index_k';
    end
end
