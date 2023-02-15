%{
README.txt:
这是一个KNN分类器

输入：train, test, train_labels, test_labels
train: 每列是一个数据
test: 每列是一个数据
train_lebals, test_labels: 标签，字典形式。 c*N维矩阵
%}

function acc_rate = KNN(train, test, train_labels, test_labels, K)
    d = size(train, 1);
    N_train = size(train, 2);
    N_test = size(test, 2);
    c = size(train_labels, 1);
    
    E = zeros(1, N_train);
    acc = 0;
    
    for i = 1:N_test
        %算欧式距离
        for j = 1:N_train
            E(j) = norm(train(:,j) - test(:,i));
        end
        %升序排序
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