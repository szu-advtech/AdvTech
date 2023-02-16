file_dir = './data/AR120p20s50by40.mat';
data_mat = load(file_dir).AR120p20s50by40;

size(data_mat);

class_nums = 120;
num = 20;
train_nums = 5;
labeled_nums = 2;
max_iter = 10;  %最大迭代次数
k = 50;  %k近邻参数 
k_pca = 200;  
alpha = 0.7;  %正则项系数
repeatN = 20; %重复实验次数


[heigth, width, ~] = size(data_mat);


acc_rate = 0;
for repeat_i = 1:repeatN  
    %{
    function [X_train, X_test, train_labels, test_labels] = 
    get_X_and_labels(data_mat, class_nums, num, train_nums, width, heigth)
    %}
    [X_train, X_test, Y_train, Y_test] = get_X_and_labels(data_mat, class_nums, ...
        num, train_nums, width, heigth);
    %size(X_train)==d*N
    %size(Y_train)==c*N
    
    %impose PCA on original data
    P_pca = PCA(X_train, k_pca);
    X_train = P_pca'*X_train;
    X_test = P_pca'*X_test;
    
    %
    [X, XL, XU, Y, YL, YU] = get_L_and_U(X_train, Y_train, ...
        train_nums, labeled_nums, class_nums);
    %size(XL)==labeled_nums*class_nums
    %size(XU)==(num-labeled_nums)*class_nums   
    
    %Initalize P by Ridge Regression
    P_RR = RR(XL, YL);
    P = P_RR;
    %size(P)==d*c
    
    %solve the optimization problem interativly
    for iter = 1:max_iter
        %{
        YU_predict = func_dpdateY(P, XU);
        Y  = [YL, YU_predict];
        %}
    
        %update W
        W = func_updateW(P, X, k);
        %update P
        Dp = func_updateDp(P);

        P = func_updateP(W, Y, X, Dp, alpha);
    
        YU_predict = func_dpdateY(P, XU);
        Y  = [YL, YU_predict];
    end    
    
    %predict
    Y_predict = P'*X_test;
    mid_Y = zeros(size(Y_predict));
    [~, c_index] = max(Y_predict);
    for i =1:size(X_test, 2)
        mid_Y(c_index(i), i)=1;
    end
    Y_predict = mid_Y;
      
    %compute accuracy
    acc_num = 0;
    for n = 1:size(Y_predict, 2)
        if Y_predict(:, n) == Y_test(:, n)
            acc_num = acc_num + 1;
        end
    end
    acc_i = acc_num/size(Y_predict, 2);
    acc_rate = acc_rate + acc_i;

end

acc_rate = acc_rate/repeatN;
disp(['Train_nums = ', num2str(train_nums), ', labeled_nums = ', num2str(labeled_nums), ', accuracy=', num2str(acc_rate)])



