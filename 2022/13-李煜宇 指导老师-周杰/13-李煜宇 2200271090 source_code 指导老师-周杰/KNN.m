%{
README.txt:
����һ��KNN������

���룺train, test, train_labels, test_labels
train: ÿ����һ������
test: ÿ����һ������
train_lebals, test_labels: ��ǩ���ֵ���ʽ�� c*Nά����
%}

function acc_rate = KNN(train, test, train_labels, test_labels, K)
    d = size(train, 1);
    N_train = size(train, 2);
    N_test = size(test, 2);
    c = size(train_labels, 1);
    
    E = zeros(1, N_train);
    acc = 0;
    
    for i = 1:N_test
        %��ŷʽ����
        for j = 1:N_train
            E(j) = norm(train(:,j) - test(:,i));
        end
        %��������
        [~, ind] = sort(E, 'ascend');
        
        
        count_class = zeros(1, c);
        for j = 1:K
            recg = find(train_labels(:, ind(j)), 1, 'first');
            count_class(recg) =  count_class(recg) + 1;
        end
        [~, recg] = max(count_class);
       
        if (recg == find(test_labels(:, i), 1, 'first'))
            acc = acc + 1;
        end
        acc_rate = acc/N_test;
    end
end