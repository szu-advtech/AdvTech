% 出处 https://github.com/kunzhan/GSF
% 作者个人主页 https://github.com/kunzhan
function [nc_eigVector, eigValue_all] = getEig(A, numOfCluster)

A = max(A,A');

% d是一个对角阵。每个元素都是A的特征值
% eigVector_matrix是A的特征向量组成的矩阵。矩阵的第i列是A的第i个特征值对应的特征向量
[eigVector_matrix, eigValue] = eig(A);

eigValue = diag(eigValue);                  % 用列向量表示特征值，eigValue是一个列向量

[~, index] = sort(eigValue);                % 从小到大排序特征值，index是一个向量，它表示排序后的元素在原数组的位置

index1 = index(1:numOfCluster);             % 取出前nc个特征值的下标
nc_eigVector = eigVector_matrix(:,index1);  % 取出前nc个特征值的特征向量
eigValue_all = eigValue(index);             % 取出所有的特征值(排序后)