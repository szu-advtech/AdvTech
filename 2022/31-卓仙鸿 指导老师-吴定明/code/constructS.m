% 出处 https://github.com/kunzhan/GSF
% 作者个人主页 https://github.com/kunzhan
function S = constructS(dataOfEachView, numOfNeighbors)

% 用k-NN算法构造每个视图的similarity matrix
% similarity matrix的每一列根据（23）式来构造
% numOfNeighbors即论文4.2中的k，也是（23）式的k
% dataOfEachView的每一列是一个图片

[~, numOfData] = size(dataOfEachView);                  % 得到该视图的图片数量
S = zeros(numOfData);                                   % 初始化similarity matrix

D = Euclidean_distance(dataOfEachView, dataOfEachView); % 与"自己"的距离矩阵

[~, index] = sort(D, 2);                                % 按距离从小到大排序

for i = 1:numOfData
    nearest = index(i,2:numOfNeighbors+2);              % k个距离最近的结点，从2开始是因为"1"是"自己"
    bj = D(i, nearest);                                 % （23）式的bij
    S(i,nearest) = (bj(numOfNeighbors+1)-bj)/(numOfNeighbors*bj(numOfNeighbors+1)-sum(bj(1:numOfNeighbors))); % （23）式
end

S = (S+S')/2;



