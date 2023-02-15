function [label] = func_kNN(data,data_ori,label_ori,k)
%% k���ڷ�����
% Input: data �������������� d * n1
%        data_ori �ѷ����������� d * n2
%        label_ori �ѷ���������ǩ 1 * n2
%        k ���������ͨ��Ϊ1��3��5
% output: label ������ǩ

%% Function body
[~,n1] = size(data);
[~,n2] = size(data_ori);
label = zeros(1,n1);
for i = 1:n1
    norm_k = zeros(1,k);
    index_k = zeros(1,k);
    for j = 1:k
        norm_k(j) = Inf;
    end
    for j = 1:n2
        if (i~=j)
            eij = norm(data(:,i)-data_ori(:,j));
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
%     index_k = uint16(index_k);
    label_k = label_ori(index_k);
    this_label = mode(label_k);
    label(i) = this_label;
end
end