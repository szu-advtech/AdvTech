file_dir = './data/Yale5040165.mat';
data_mat = load(file_dir).Yale5040165;


class_nums = 15;
num = 11;
train_nums = 6;
labeled_nums = 2;


max_iter = 10;  %最大迭代次数
k = 30;  %k近邻参数 
k_pca = 300;  
alpha = 0.9;  %正则项系数
beta = 0.1; %L21正则项系数
gamma =1e-4;%capped正则项系数
epsilon = 1e9;
repeatN = 10; %重复实验次数


[heigth, width, ~] = size(data_mat);

path = cd;   %获取当前路径
pic = 'loss_carve'; %将字符型变量赋予pic
mkdir(pic) %创建名为‘图片库’的文件夹
row_count = 1;
par_list = {' ',  'My_algorithm', 'Ridge_Regression', 'SSMMC', 'SELF', 'PNLP', 'ALPMTR'};
writecell(par_list,'Yale_results.xls','Sheet',1,'Range','A1:G1');



acc_rate = [];
rr_acc_rate = [];
SSMMC_acc_rate = [];
SELF_acc_rate = [];
PNLP_acc_rate = [];
ALPMTR_acc_rate = [];   
for repeat_i = 1:repeatN  
    %{
    function [X_train, X_test, train_labels, test_labels] = 
    get_X_and_labels(data_mat, class_nums, num, train_nums, width, heigth)
    %}
    data_mat = func_digahole(data_mat, 0);
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
    [X, XL, XU, Y, YL, YU, YU_label] = get_L_and_U(X_train, Y_train, ...
        train_nums, labeled_nums, class_nums);
    %size(XL)==labeled_nums*class_nums
    %size(XU)==(num-labeled_nums)*class_nums   
   
    %Initalize P by Ridge Regression
    P_RR = RR(XL, YL);
    %P = P_RR;
    %size(P)==d*c
    
    %solve the optimization problem interativly
    %{
    for iter = 1:max_iter
        %{
        YU_predict = func_dpdateY(P, XU);
        Y  = [YL, YU_predict];
        %}
        %update Y
        YU_predict = func_dpdateY(P, XU);
        Y  = [YL, YU_predict];
        Dr = func_updateDr(Y, P, X);

        %update XD Dp
        XD = func_updateXD(X, Y);
        DD = func_updateDD(XD, P, epsilon);


        %update W
        W = func_updateW(P, X, k);

        %update P
        Dp = func_updateDp(P);

        P = func_updateP(W, Y, X, XD, DD, Dp, Dr, alpha, beta, gamma);  
        
    end    
    %}
    
    alpha = 0.8;  %正则项系数
    beta = 0.1; %L21正则项系数
    gamma =1e-4;%capped正则项系数
    epsilon = 1e9;
    P = func_my_algorithm(XL, XU, YL, max_iter, k, alpha, beta, gamma, epsilon);

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
    acc_rate_i = acc_num/size(Y_predict, 2);
    acc_rate = [acc_rate, acc_rate_i];


    %RR
    Y_RR_pre = P_RR'*X_test;
     
    mid_Y = zeros(size(Y_RR_pre));
    [~, c_index] = max(Y_RR_pre);
    for i =1:size(X_test, 2)
        mid_Y(c_index(i), i)=1;
    end
    Y_RR_predict = mid_Y;
    
 
    rr_acc_num = 0;
    for n = 1:size(Y_RR_pre, 2)
        if Y_RR_predict(:, n) == Y_test(:, n)
            rr_acc_num = rr_acc_num + 1;
        end
    end
    rr_acc_i = rr_acc_num/size(Y_RR_predict, 2);
    rr_acc_rate = [rr_acc_rate, rr_acc_i];

    %SSMMC  
    SSMMC_sigma = 2e4;
    SSMMC_lamda1 = 1e1;
    SSMMC_lamda2 = 1e1;
    P_SSMMC = func_SSMMC(XL, XU, YL, 100, SSMMC_sigma, SSMMC_lamda1, SSMMC_lamda2);
    SSMMC_acc_rate_i = KNN(P_SSMMC'*XL, P_SSMMC'*X_test, YL, Y_test, 1);
    SSMMC_acc_rate = [SSMMC_acc_rate, SSMMC_acc_rate_i];

    %SELF
    for i = 1:size(YL, 2)
        [~, yl] = max(YL);
    end
    y=[yl,zeros(1, size(YU, 2))];
    [T,Z]=SELF(X,y, 0.5, 120);    
    SELF_acc_rate_i = KNN(T'*X_train, T'*X_test, Y_train, Y_test, 1);
    SELF_acc_rate = [SELF_acc_rate, SELF_acc_rate_i];
    %}

    %PNLP
    PNLP_sigma =1e3;
    PNLP_miu = 10;
    PNLP_miu1 = 0.3;
    PNLP_miu2 = 0.7;
    YU_for_PNLP = zeros(class_nums, class_nums*(train_nums-labeled_nums));
    size(YL);
    size(YU);
    F = func_PNLP(XL, XU, YL, YU_for_PNLP, PNLP_sigma, PNLP_miu, PNLP_miu1, PNLP_miu2);
    
    
    %predict
    F = F';
    PNLP_acc_num = 0;
    aux_X = [XL, XU];
    for i = 1:size(X_test, 2)
        aux_sum = 0;
        F_sum = zeros(class_nums, 1);
        for j = 1:size(aux_X, 2)

            Wxxj = exp(-norm(X_test(:,i)-aux_X(:, j))^2/(sigma^2));
            aux_sum = aux_sum + Wxxj;
            F_sum = F_sum + Wxxj*F(:, j);
        end
        f_predict_i = 1/(aux_sum - PNLP_miu*PNLP_miu2)*(F_sum - PNLP_miu*PNLP_miu2*ones(class_nums, 1));
        [~, predict_i] = max(f_predict_i);
        [~, ture_i] = max(Y_test(:, i));
        if predict_i == ture_i
            PNLP_acc_num = PNLP_acc_num + 1;
        end

    end
    PNLP_acc_rate_i = PNLP_acc_num/size(Y_test, 2);
    PNLP_acc_rate = [PNLP_acc_rate, PNLP_acc_rate_i];


    %ALP-TMR
    
    ALPMTR_alpha = 10000;
    ALPMTR_beta = 10000;
    ALPMTR_gamma = 5e2;
    YU_0 = zeros(class_nums, class_nums*(num-labeled_nums));  
    [F, W] = func_ALPTMR(XL, [XU, X_test], YL, YU_0, ALPMTR_alpha, ALPMTR_beta, ALPMTR_gamma);
     %predict
    Y_predict = F(:, class_nums*labeled_nums+1:size(F, 2));
    mid_Y = zeros(size(Y_predict));
    [~, c_index] = max(Y_predict);
    for i =1:size(X_test, 2)
        mid_Y(c_index(i), i)=1;
    end
    Y_predict = mid_Y;
      
    %compute accuracy
    Y_test = [YU_label, Y_test];
    acc_num = 0;
    for n = 1:size(Y_predict, 2)
        if Y_predict(:, n) == Y_test(:, n)
            acc_num = acc_num + 1;
        end
    end
    ALPMTR_acc_rate_i = acc_num/size(Y_predict, 2);
    ALPMTR_acc_rate = [ALPMTR_acc_rate, ALPMTR_acc_rate_i];
    %}

end


acc_rate_mean = mean(acc_rate);
rr_acc_rate_mean = mean(rr_acc_rate);
SSMMC_acc_rate_mean = mean(SSMMC_acc_rate);
SELF_acc_rate_mean = mean(SELF_acc_rate);
PNLP_acc_rate_mean = mean(PNLP_acc_rate);
ALPMTR_acc_rate_mean = mean(ALPMTR_acc_rate);

acc_rate_std = std(acc_rate);
rr_acc_rate_std = std(rr_acc_rate);
SSMMC_acc_rate_std = std(SSMMC_acc_rate);
SELF_acc_rate_std = std(SELF_acc_rate);
PNLP_acc_rate_std = std(PNLP_acc_rate);
ALPMTR_acc_rate_std = std(ALPMTR_acc_rate);

acc_rate_max = max(acc_rate);
rr_acc_rate_max = max(rr_acc_rate);
SSMMC_acc_rate_max = max(SSMMC_acc_rate);
SELF_acc_rate_max = max(SELF_acc_rate);
PNLP_acc_rate_max = max(PNLP_acc_rate);
ALPMTR_acc_rate_max = max(ALPMTR_acc_rate);

acc_rate_min = min(acc_rate);
rr_acc_rate_min = min(rr_acc_rate);
SSMMC_acc_rate_min = min(SSMMC_acc_rate);
SELF_acc_rate_min = min(SELF_acc_rate);
PNLP_acc_rate_min = min(PNLP_acc_rate);
ALPMTR_acc_rate_min = min(ALPMTR_acc_rate);

result_list = {'means',  num2str(acc_rate_mean), num2str(rr_acc_rate_mean), num2str(SSMMC_acc_rate_mean), num2str(SELF_acc_rate_mean), num2str(PNLP_acc_rate_mean), num2str(ALPMTR_acc_rate_mean)};
writecell(result_list,'Yale_results.xls','Sheet',1,'Range','A2:G2');
result_list = {'std',  num2str(acc_rate_std), num2str(rr_acc_rate_std), num2str(SSMMC_acc_rate_std), num2str(SELF_acc_rate_std), num2str(PNLP_acc_rate_std), num2str(ALPMTR_acc_rate_std)};
writecell(result_list,'Yale_results.xls','Sheet',1,'Range','A3:G3');

result_list = {'max',  num2str(acc_rate_max), num2str(rr_acc_rate_max), num2str(SSMMC_acc_rate_max), num2str(SELF_acc_rate_max), num2str(PNLP_acc_rate_max), num2str(ALPMTR_acc_rate_max)};
writecell(result_list,'Yale_results.xls','Sheet',1,'Range','A4:G4');

result_list = {'min',  num2str(acc_rate_min), num2str(rr_acc_rate_min), num2str(SSMMC_acc_rate_min), num2str(SELF_acc_rate_min), num2str(PNLP_acc_rate_min), num2str(ALPMTR_acc_rate_min)};
writecell(result_list,'Yale_results.xls','Sheet',1,'Range','A5:G5');

disp(['Alpha = ', num2str(alpha), ', train_nums = ', num2str(train_nums), ', labeled_nums = ', num2str(labeled_nums), ', accuracy=', num2str(acc_rate_mean), ', RR_accuracy=', num2str(rr_acc_rate_mean), ', SSMMC_accuracy=', num2str(SSMMC_acc_rate_mean), ', SELF_accuracy=', num2str(SELF_acc_rate_mean), ', PNLP_acc_rate=', num2str(PNLP_acc_rate_mean), ' ,ALPMTR_acc_rate=', num2str(ALPMTR_acc_rate_mean)])

