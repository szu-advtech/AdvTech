file_dir = './data/AR120p20s50by40.mat';
data_mat = load(file_dir).AR120p20s50by40;

size(data_mat);

class_nums = 120;
num = 20;
train_nums = 3;

[heigth, width, ~] = size(data_mat);

%{
function [X_train, X_test, train_labels, test_labels] = 
get_X_and_labels(data_mat, class_nums, num, train_nums, width, heigth)
%}
[X_train, X_test, Y_train, Y_test] = get_X_and_labels(data_mat, class_nums, ...
    num, train_nums, width, heigth);

P_pca = PCA(X_train, k_pca);

X_train = P_pca'*X_train;
X_test = P_pca'*X_test;

P_RR = RR(X_train, Y_train);

Y_pre = P_RR'*X_test;


mid_Y = zeros(size(Y_pre));
[~, c_index] = max(Y_pre);
for i =1:size(X_test, 2)
    mid_Y(c_index(i), i)=1;
end

Y_predict = mid_Y;



acc_num = 0;

for n = 1:size(Y_pre, 2)
    if Y_predict(:, n) == Y_test(:, n)
        acc_num = acc_num + 1;
    end

end
rr_acc_rate = acc_num/size(Y_predict, 2)