% Parent = replace_decision_chromosome_kmeans(temp_pop,n_obj,n_var, NP);
% temp_pop intermediate_chromosome ： 400*8，子代也经过非支配kmeans排序后得到的
% NP pop ：200
% M=2，V=2，pop=200
function f = replace_decision_chromosome_kmeans(intermediate_chromosome, M, V, pop)

% 按x1降序排列，选前几个
[ C,IA,IC ] = unique(intermediate_chromosome(:,1:V),'rows'); % unique是获得矩阵中非重复的行，基于前两列的数据查找intermediate_chromosome表中的唯一行（即没重复的数，如[1 2]和[1 2]算重复，只记一次），并按第1列升序排序。C是得到的第1列排序后的数据，IA是对应C第一列的索引，IC是对应intermediate_chromosome表第一列的索引
intermediate_chromosome = intermediate_chromosome(IA,:);  % 是按第1列升序排序后的数据（400*8）
% IA为矩阵C（去掉重复元素的矩阵）中的元素在矩阵A中的位置，（intermediate_chromosome是A）
% IC为矩阵A中的元素在矩阵C中的位置，即C中哪几行（索引）构成了A。
[N, m] = size(intermediate_chromosome); % N=400，m=8
% Get the index for the population sort based on the rank 按等级（第五列）排序
[temp,index] = sort(intermediate_chromosome(:,M + V + 1)); 
clear temp m

% 现在根据等级索引对个体进行排序，得到按等级排序的400个个体（这里面包括父代和子代的8列数据）
for i = 1 : N
    sorted_chromosome(i,:) = intermediate_chromosome(index(i),:);
end

% Find the maximum rank in the current population 找出当前种群中的最大等级，如22
max_rank = max(intermediate_chromosome(:,M + V + 1));

% 开始添加基于等级和拥挤度距离的每个前沿，直到整个种群被填满。
previous_index = 0;
for i = 1 : max_rank
    % Get the index for current rank i.e the last the last element in the
    % sorted_chromosome with rank i. 得到当前等级为i的个体数
    current_index = max(find(sorted_chromosome(:,M + V + 1) == i)); % find找等级为1的个体的索引（find得到81*1，即1-81连续数，即sorted_chromosome表前81都是等级为1的个体）。i=2时，current_index值为前两个等级个体数相加
    
    % 如果所有等级i的个体都被添加到种群中，检查种群是否被填充。
    if current_index > pop  % 如果当前等级个体数大于200
        % 如果是，那么找出前几个等级的总个体数
        remaining = pop - previous_index; % 200减167（前三个等级的个体总数），算出离容器容量200个还差多少个个体
        temp_pop = ...
            sorted_chromosome(previous_index + 1 : current_index, :);  % 表里的168-207行（第四等级的个体）赋值给temp_pop表
        [temp_sort,temp_sort_index] = ...
            sort(temp_pop(:, M + V + 3),'descend');  % 把第4等级个体（因为要取第4等级里最优秀的个体到容器里，构成200个个体）按第7列(特殊拥挤度距离)降序排列得到temp_sort
        for j = 1 : remaining  % j=1:33
            f(previous_index + j,:) = temp_pop(temp_sort_index(j),:); %在f的167列（前167列包括前三个等级的个体）往后添加第四等级按特殊拥挤度距离降序排列的前33个个体
        end
        return;
    elseif current_index < pop  % 如果当前等级个体数小于200，继续添加
        % 把所有等级为i的个体加到种群中（一个等级一个等级地加）
        f(previous_index + 1 : current_index, :) = ...
            sorted_chromosome(previous_index + 1 : current_index, :); %一次把等级i的个体放到f里（累加），把第一等级个体(1-81行)弄到f表里，i=2时，把第二等级个体(82-127行)也加入到f表里的82-127行
    else
        % 将所有等级为i的个体加入种群
        f(previous_index + 1 : current_index, :) = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
        return;
    end
    % 获取最后添加的个体的索引（即1-81取81，82-127取127）
    previous_index = current_index;  % 把第一等级个体数赋值给previous_index，i=2时，current_index值为前两个等级个体数相加
end
end


