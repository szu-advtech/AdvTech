% 出处 https://github.com/kunzhan/GSF
% 作者个人主页 https://github.com/kunzhan
function [ACC,NMI,Purity,Precision,Recall,F_score,ARI,cluster_num,kmeans_value] = GSF(S_init,numOfCluster,groundtruth,gamma1,gamma2 ,adaptive_neighbor)

numOfView = size(S_init,3);     % 视图的个数
numOfData = size(S_init,1);     % 图片的个数

A = ones(numOfData);            % 初始化（1）式中的A
for i = 1:numOfView
    % Hadamard积
    A = A-diag(diag(A));        % 让对角线全0
    A = A.*S_init(:,:,i);       % 将所有视图用Hadamard积融合起来，得到（1）式的A
end

% 下面5条语句其实是计算A的拉普拉斯矩阵的
% 参考论文的（2）式的下面那段说明
% 经过 A = (A + A')/2; 后，对diag(sum(A))的值没有影响
A = (A + A')/2;                           % （1）式中的A
Diagram_A = diag(sum(A));                 
Laplacian_A = Diagram_A - A;              % A的拉普拉斯矩阵
[U, ~]=getEig(Laplacian_A, numOfCluster); % 融合后的亲和矩阵A的特征向量
A_temp = A;

distance_U_First = Euclidean_distance(U',U');   % 迭代开始前，需要先构造一个初始的S，而这个S根据A算出来的U决定
[~,index] = sort(distance_U_First,2);           % 按距离从小到大排列矩阵U的每个特征向量


% 下面这个循环就是论文的Algorithm 1
% 下面的循环是想得到一个矩阵S。这个S刚好代表有nc个连通分量
% 然而，矩阵S的每一列Sj是要根据Eq.(13)得到的，那么我们必须要知道γ2是多少
% 我们只是给了一个γ2的初始值。然后慢慢找到正确的γ2，最后更新出S
% 所以，下面这个循环主要内容是：1.用Eq.(13)更新S的每一列，最后得到一个S。
% 2. 求出新的S的拉普拉斯矩阵L。3. 算L前nc个特征值的和。如果为0，说明我们找到了gamma2，否则继续迭代
% 4. 如果找到γ2，退出循环，即可得到最终的S，S恰好有nc个连通分量！
iteration_time = 30;
objective_value = [];                                   % 用来画出论文figure 3的
for i = 1:iteration_time
    % 迭代S
    S = zeros(numOfData);                               % 初始化要迭代的S
    distance_U2 = Euclidean_distance(U',U');            % 因为要根据ui-uj更新S，所以每次都要计算一遍
    
    for j=1:numOfData
        index_a0 = index(j,2:adaptive_neighbor + 1);    % 最近的adaptive_neighbor个结点的位置。从2开始是因为"1"是结点本身
        % (10)-(13)式
        ui_subtract_uj = distance_U2(j,index_a0);       % |ui - uj|^2
        aij = A_temp(j,index_a0);                       % aij
        pij = ui_subtract_uj - gamma1*aij;              % pij
        sj_star = -pij/(2*gamma2);                      % (13)式

        S(j,index_a0) = EProjSimplex_new(sj_star);      % 由sj_star排成的矩阵。此时一次迭代已完成  
    end
    
    % S一次迭代完成后，求S的拉普拉斯矩阵
    S = (S + S')/2;
    Diagram_S = diag(sum(S));
    Laplacian_S = Diagram_S - S;
    U_old = U;

    % 根据新的S更新矩阵U，稍后找到γ2
    [U, eigenValue]=getEig(Laplacian_S, numOfCluster);                                            % 排序后的所有的特征向量
    objective_value(i) = trace(U'*Laplacian_S*U) - gamma1*trace(S*A_temp) + gamma2*trace(S*S');   % (9)式目标函数值

    % 更新γ2的值
    % L是半正定的，特征值大于等于0，他就是要L的前nc个特征值的和为0。
    % 阈值其实没有严格为0，可以说是一个可以忍受的误差，参考论文第6页最后一段
    threshold = 1*10^-5;

    % 想想迫敛性准则
    sum_eigenValue_1 = sum(eigenValue(1:numOfCluster));
    sum_eigenValue_2 = sum(eigenValue(1:numOfCluster+1));
    
    % 根据论文5.1的最后部分更新γ2
    if sum_eigenValue_1 > threshold          % γ2太大
        gamma2 = gamma2/2;
    elseif sum_eigenValue_2 < threshold      % γ2太小
        gamma2 = gamma2*2;  
        U = U_old;
    else                                     % 找到了γ2，迭代结束，符合的S已经迭代生成
        break;
    end
end

plot(objective_value);                       % 用来画出论文figure 3的

[cluster_num, cluster_res]=graphconncomp(sparse(S));    % 直接调用matlab内置方法求出分类个数，以及分类结果
cluster_res = cluster_res';                             % 转置一下，变成列向量，之后计算聚类指标

% 参考论文5.1的最后部分。
% 重复做30次k-means方法，取最小值。
kmeans_value_store = zeros(1,30);
for i=1:30
    [~,~,temp] = kmeans(U,numOfCluster,'EmptyAction','drop');
    kmeans_value_store(i) = sum(temp);
end
kmeans_value = min(kmeans_value_store)


[ACC,NMI,Purity,Precision,Recall,F_score,ARI] = metric_compute(groundtruth, cluster_res);
