
file_dir = './data/AR120p20s50by40.mat';
data_mat = load(file_dir).AR120p20s50by40;


class_nums = 120;
num = 20;
train_nums = 15;
labeled_nums = 5;
max_iter = 10;  %最大迭代次数
knn = 30;  %k近邻参数 
k_pca = 300;  
alpha = 0.8;  %正则项系数
beta = 0.1; %L21正则项系数
gamma =0;%capped正则项系数
epsilon = 1e9;
repeatN = 5; %重复实验次数


[heigth, width, ~] = size(data_mat);

%data_mat = func_digahole(data_mat, 7);
[X_train, X_test, Y_train, Y_test] = get_X_and_labels(data_mat, class_nums, ...
        num, train_nums, width, heigth);
%size(X_train)==d*N
%size(Y_train)==c*N

%X_train = normalize(X_train, 2);

%impose PCA on original data
P_pca = PCA(X_train, k_pca);
X_train = P_pca'*X_train;
X_test = P_pca'*X_test;

%
[X, XL, XU, Y, YL, YU] = get_L_and_U(X_train, Y_train, ...
    train_nums, labeled_nums, class_nums);
%size(XL)==labeled_nums*class_nums
%size(XU)==(num-labeled_nums)*class_nums 

[P, F] = func_ProjLP(XL, XU, YL, YU, knn, 1e5);
c = size(Y, 1);

F_test = [];
for k = 1:size(X_test, 2)
    X = [X_train, X_test(:, k)];
    knn_mat = func_getKNNMat(X, knn); 
    n_total = size(X, 2);
    %W = zeros(n_total, n_total);
    all_one_vec = ones(knn, 1);
    W_i = zeros(n_total, 1);

    
    %求解Z
    A = [];
    for j= knn_mat(:, n_total)
        A = [A, X(:, n_total)-X(:, j)];
    end
    Z = A'*A;            
    Z_inv = inv(Z+0.00001*eye(knn));
    w_i = (Z_inv*all_one_vec)/(all_one_vec'*Z_inv*all_one_vec);    
    for j= 1:knn
        W_i(knn_mat(j, n_total)) = w_i(j);
    end
    size(W_i);
    size(P'*F);
    F_test = [F_test, P'*F*W_i(1:n_total-1)];
end

Y_predict = F_test;
mid_Y = zeros(size(Y_predict));
[~, c_index] = max(Y_predict);
for i =1:size(X_test, 2)
    mid_Y(c_index(i), i)=1;
end
Y_predict = mid_Y(1:c, :);
  
%compute accuracy
acc_num = 0;
for n = 1:size(Y_predict, 2)
    if Y_predict(:, n) == Y_test(:, n)
        acc_num = acc_num + 1;
    end
end
acc_rate = acc_num/size(Y_predict, 2)

