function [knn_mat] = func_getKNNMat(data,k)   %out_mat
%% ������ݵ�K���ھ���
% Input��data��ά��d*n
%        k,ȡk�����ڵ�
% Output��knn_mat��ά��k*n����ֵ�����i��������k���ڵ������±�

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
