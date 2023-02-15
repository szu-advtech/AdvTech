function Y=my_convert2Sparse_impro(X)

    mean1 = mean(X);	% �� MRI ���������ֵ
    std1 = std(X);      % �� MRI ���������׼��
    X = (X - repmat(mean1,size(X,1),1));
    Y = repmat(std1,size(X,1),1); % ÿ��ֵ����ֵ�����Ա�׼��.
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
        X = sparse(X); % ת����ϡ�����
    end
    Y=X;
end