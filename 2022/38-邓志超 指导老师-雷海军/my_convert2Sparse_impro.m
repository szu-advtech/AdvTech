function Y=my_convert2Sparse_impro(X)

    mean1 = mean(X);	% 对 MRI 所有列求均值
    std1 = std(X);      % 对 MRI 所有列求标准差
    X = (X - repmat(mean1,size(X,1),1));
    Y = repmat(std1,size(X,1),1); % 每个值减均值，除以标准差.
    length_row=size(X,1);
    length_column=size(X,2);
    for j=1:length_column
        if std1(j)==0
            continue;
        end
        for i=1:length_row
            X(i,j)=X(i,j)/Y(i,j);
        end
    end
    if ~issparse(X)
        X = sparse(X); % 转换成稀疏矩阵
    end
    Y=X;
end