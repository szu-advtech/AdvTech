% Parentstar x：里面有每个个体的两个x（决策变量）和适应度两个y（目标值）
% n_obj n_obj
% n_var n_var
% K KK：一类里有多少个点
function f = non_domination_scd_kmeans_sort(x, n_obj, n_var,KK)
% 根据非支配关系和特殊拥挤度距离来选择种群

[N_particle, ~] = size(x); % 获个体数，N_particle是200

% 初始化前沿为1
front = 1;

% 这个赋值没有任何内容，只用于在MATLAB中轻松操作。
F(front).f = []; %F是个结构体，包含多个矩阵 f。比如front=4，那么F(front).f 表示结构体中第四个矩阵。
individual = [];

%% 非支配排序（与NSGA2一样）
for i = 1 : N_particle  %1:200
    % 支配此个体的个体数量（找老大）
    % individual是结构体，包含n和p两个字段（因为每个字段里的每行不相等，不是矩阵）
    individual(i).n = 0;
    % 被它支配的个体的集合/此个体支配的个体数组（小弟）
    individual(i).p = [];
    for j = 1 : N_particle  %j = 1:200
        dom_less = 0;
        dom_equal = 0;
        dom_more = 0;
        for k = 1 : n_obj  %k = 1:2
            if (x(i,n_var + k) < x(j,n_var + k)) % if(x(1,3) < x(1,3))
                dom_less = dom_less + 1;
            elseif (x(i,n_var + k) == x(j,n_var + k)) % if(x(1,3) == x(1,3))
                dom_equal = dom_equal + 1;
            else
                dom_more = dom_more + 1;
            end
        end
        % 该个体没有一项是强于另一个个体的(在数值上表现是小于)，且不是每一个目标项都相等
        if dom_less == 0 && dom_equal ~= n_obj
            individual(i).n = individual(i).n + 1;
        % 该个体没有一项是弱于另一个个体的（数值上表现是大于），且不是每一个目标都相等
        elseif dom_more == 0 && dom_equal ~= n_obj
            individual(i).p = [individual(i).p j];
        end
    end
    % 如果支配此个体的个体数量为0的话，说明无其他个体支配此个体
    if individual(i).n == 0
        % 将第五列作为rank等级 第一等级为1
        x(i,n_obj + n_var + 1) = 1;
        % F(front).f是当前（第一个）的前沿个体的集合，把编号i放入到F的f里
        F(front).f = [F(front).f i];
    end
end
% 查找后续前沿，设置等级
while ~isempty(F(front).f)  % 表示如果F(front).f是空元素，结果为0（false）。不空就执行，F(front).f是除了上一前沿的
    Q = [];
    for i = 1 : length(F(front).f)   % 第一个前沿的个体数
        if ~isempty(individual(F(front).f(i)).p)   % F(1).f(1)是第一个前沿的第一个个体编号， (individual(2).p)是第一个前沿的第一个个体（编号不一定为1）的所有小弟
            % 将individual(i).p的个体的支配等级-1
            for j = 1 : length(individual(F(front).f(i)).p)
                individual(individual(F(front).f(i)).p(j)).n = ...
                    individual(individual(F(front).f(i)).p(j)).n - 1;
                % 然后将新的为0的个体的rank等级设置为front+1，新的为0的个体加入到前沿中
                if individual(individual(F(front).f(i)).p(j)).n == 0
                    x(individual(F(front).f(i)).p(j),n_obj + n_var + 1) = ...
                        front + 1;
                    % individual(F(front).f(i)).p(j)含义：第front次前沿的第fi个个体的第pj个支配的个体
                    Q = [Q individual(F(front).f(i)).p(j)];
                end
            end
        end
    end
    front =  front + 1;
    F(front).f = Q;   % F结构体中存放各个等级的个体，第一行代表等级为1的个体，共22行，即有1-22等级
end



% 根据所处前沿等级数对种群进行排序
[~,index_of_fronts] = sort(x(:,n_obj + n_var + 1));  %x(:,5)是等级列，给所有个体按等级排序，index_of_fronts是按照等级排序后得到的对应200个体的编号
for i = 1 : length(index_of_fronts)   %i = 1 : 200
    sorted_based_on_front(i,:) = x(index_of_fronts(i),:);   %把x表格按第五列等级排序变成sorted_based_on_front表格
end
% 在每一个前沿面执行k-means，然后计算决策空间距离
finalpop = [];
avg_crowd_dist_var = [];
%% 遍历各个帕累托等级
% 在每个等级中聚类，分别按照决策空间两个变量计算拥挤度距离，然后求平均作为个体真正的拥挤度距离
for front = 1 : (length(F)-1)  % front = 1 : 21，最后一个前沿等级是空的，所以要减一
    front_index = find(sorted_based_on_front(:,n_obj + n_var + 1)==front);   % front_index (26*1)是指前沿为front（等级为1）的个体在sorted_based_on_front表里对应的编号（不是个体编号哦），find返回非零元素下标
    zhongjianpop = sorted_based_on_front(front_index,:);    % zhongjianpop 是指 rank为front（等级为1）的个体所有数据（21*5）
    K = ceil(length(front_index)/KK);  %向上取整，length(front_index)等级为1的个体有26个，KK相当于将约10个点聚为一类，一共聚为K类
    if size(zhongjianpop,1) < K
        keyboard
    end
    % 将个体聚类为K类（根据决策变量x1和x2） ind是指每个个体对应的聚类类别编号，如果K=29/10=3，就有编号1 2 3
    if size(zhongjianpop,1) > 1
        X=zscore(zhongjianpop(:,1:n_var)); % 标准化数据
%         fprintf("%d====%d----%d \n", front, length(F)-1, size(zhongjianpop,1))
        Y=pdist(X); % 计算距离，Pdist函数用于各种距离的生成
        Z=linkage(Y); %定义变量之间的连接，使用指定的method创建树
    %    C=cophenet(Z,Y); %评价聚类信息
        ind=cluster(Z,K); %创建聚类，从聚类树结构中构建聚类
    else
        ind=[1];
    end

%     [ind,~] = kmeans(zhongjianpop(:,1:n_var),K);
%     X=mapminmax(zhongjianpop(:,1:n_var),0,1); % 'mapminmax' 需要 Deep Learning Toolbox。
%     T1=clusterdata(X,0.2);

%     Z = linkage(zhongjianpop(:,1:n_var), 'ward');
%     ind = fcluster(Z,4,'distance');
%     ind = clusterdata(zhongjianpop(:,1:n_var),'Linkage','ward','SaveMemory','on','Maxclust',4);
    pop=cell(max(ind),1);    % cell 一种数据类型，包含max(ind)=3个空矩阵，为了分别存放类别1 2 3的个体的数据（五列：x1 x2 f1 f2 front），类别1有16*5。
    % 将第一等级的所有个体归到对应的类中(聚类)，遍历第一等级所有个体根据决策变量划分三类
    for i1=1:length(ind)   % i1=1:26，把3个类存到pop里对应的矩阵中，第一个矩阵存放第一类。pop{3,1}/pop{3}代表类别3的所有个体
        pop{ind(i1)} = [pop{ind(i1)};zhongjianpop(i1,:)];
    end
    % 遍历每一个聚类，在每一个聚类中计算决策空间的拥挤度距离
    for i2= 1 : size(pop,1)  % i2= 1 : 3（聚成3类）
        crowd_dist_obj = 0;
        y = pop{i2};  % 当i2=1时，聚类第一个类别所有个体（16*5），赋值给y，即y是第一个类别的个体
        sorted_based_on_objective = []; %一个空数组
        for i = 1 : n_var   % i=1:2，遍历第一等级个体的决策空间两个变量
            [sorted_based_on_objective, index_of_objectives] = ...
                sort(y(:,i));   % 得到 把y按照决策空间的第一个变量x1从小到大排序（16*1） 及 对应的索引（16*1）（即对应y表的行数）
            sorted_based_on_objective = []; % 清零
            for j = 1 : length(index_of_objectives) % j=1:16（第一等级中第一类的个体数）
                sorted_based_on_objective(j,:) = y(index_of_objectives(j),:); % 按照索引排列，即按照决策变量x1进行从小到大排列，这个操作是：y的第14行所有列（5列）赋值给sorted_based_on_objective表的1行
            end
            % 获取所有个体决策变量x1的最大/小值（始终是对某一个front的某一个聚类来说）
            f_max = ...
                sorted_based_on_objective(length(index_of_objectives), i); %length(index_of_objectives)是第一前沿第一类的总个体数，最大值是第一列中的最后一个（因为已经排好序了）
            f_min = sorted_based_on_objective(1,  i); %最小值是第一列中的第一个

            % 先计算决策空间的边界个体的拥挤度距离：没有邻居/只有一个邻居
            % 某一聚类中只有一个个体的特殊情况：每一个决策空间对应的拥挤度距离置为1
            if length(index_of_objectives)==1 % 如果第一类只有1个个体，index_of_objectives存放第一类所有个体
                y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;  % index_of_objectives(1)是9，把y(未按x1排序)的第9行（对应按决策空间x1从小到大排序后第一个个体）第6列（拥挤度距离）设为1
            % 某一聚类中只有两个个体的特殊情况：每一个决策空间对应的拥挤度距离置为1
            elseif length(index_of_objectives)==2
                y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;
                y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)=1; % y的第4行（对应按决策空间x1从小到大排序后最后一个个体）第6列（拥挤度距离）设为1
            else
                % 否则计算决策空间的边缘个体（最大x1所对应的个体和最小x1所对应的个体）的拥挤度距离，计算方式：2 *（邻居某变量-该个体某变量）/ （f_max - f_min）
                y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)...
                    = 2*(sorted_based_on_objective(length(index_of_objectives), i)-...
                    sorted_based_on_objective(length(index_of_objectives) -1, i))/(f_max - f_min);  %先从最大的x1来算，放到index_of_objectives的第6列作为第一类最大x1所在的个体的拥挤度距离，如何计算？ 2*(最大的x1-次大的x1)/(f_max - f_min)
                y(index_of_objectives(1),n_obj + n_var + 1 + i)=2*(sorted_based_on_objective(2, i)-...
                    sorted_based_on_objective(1, i))/(f_max - f_min);  %先从最小的x1来算，如何计算最小x1的拥挤度距离？ 2*(最小的x1-次小的x1)/(f_max - f_min)
            end
            % 中间个体的拥挤度距离计算方式：(next_obj - previous_obj)/(f_max - f_min)
            for j = 2 : length(index_of_objectives) - 1 %j=2:(11-1)，遍历第二个个体到倒数第二个个体
                next_obj  = sorted_based_on_objective(j + 1, i); %sorted_based_on_objective表（x1按从大到小排列）的3行1列
                previous_obj  = sorted_based_on_objective(j - 1,i); %sorted_based_on_objective表的1行1列
                if (f_max - f_min == 0) % 如果决策变量x1的最大值等于最小值
                    y(index_of_objectives(j),n_obj + n_var + 1 + i) = 1; %index_of_objectives(2)是7，即y的第7行，即x1第二小的个体，第5列（拥挤度）设为1
                else
                    y(index_of_objectives(j),n_obj + n_var + 1 + i) = ...
                        (next_obj - previous_obj)/(f_max - f_min); % (第3个个体的x1-第1个个体的x1)/(f_max - f_min)，算出按x1从小到大排序第二行个体的拥挤度
                end
            end
        end
        %% 计算决策空间拥挤度距离（之前分别算了两个变量的拥挤度距离，现在平均一下）
        crowd_dist_var = [];
        crowd_dist_var(:,1) = zeros(size(pop{i2},1),1); %6*1，即第一类有6个个体
        % 个体的拥挤度距离 = 个体每一个变量拥挤度距离的和除以变量总数（即平均）
        for ii = 1 : n_var  %ii = 1 : 2
            crowd_dist_var(:,1) = crowd_dist_var(:,1) + y(:,n_obj + n_var + 1 + ii); % 0+第6列(按x1算出的拥挤度距离)的数字 赋值 到 crowd_dist_var表中第1列。crowd_dist_var表中第1列的数 + 第7列(按x2算出的拥挤度距离)的数字 还是加到（即叠加） crowd_dist_var表中第1列.即两个决策变量的拥挤度距离相加
        end
        crowd_dist_var=crowd_dist_var./n_var; %两个决策变量的拥挤度距离相加 再除以 决策变量个数
        %avg_crowd_dist_var_k=mean(crowd_dist_var); %再平均
        y(:,n_obj+n_var+2)=crowd_dist_var;    % n_obj+n_var+2置决策空间的拥挤度距离指标，把两个决策变量的拥挤度距离相加 再除以 决策变量个数 的数 覆盖y表的第6列（原本y表的第6列是按x1算出的拥挤度距离）
        y = y(:,1 : n_obj + n_var+2 );  % 将多余的过程指标去除(冗余指标)，把第7列（按x2算出的拥挤度距离）去掉
        % 将产生决策空间拥挤度距离的个体放到finalpop中作为最终得到的种群
        finalpop=[finalpop;y]; % finalpop与y一样
       % avg_crowd_dist_var(front) = [avg_crowd_dist_var(front);avg_crowd_dist_var_k];
    end
end

sorted_based_on_front = [];
%% 计算目标空间的距离
[~,index_of_fronts] = sort(finalpop(:,n_obj + n_var + 1)); %按第5列（等级）排序（实际上已经排好序了）
for i = 1 : length(index_of_fronts)  % i=1:200
    sorted_based_on_front(i,:) = finalpop(index_of_fronts(i),:);
   % avg_crowd_dist_var(i,:) = avg_crowd_dist_var(index_of_fronts(i),:);
end
  current_index = 0; %记录当前个体的索引，如果把第一等级个体遍历完，那么该值为第一等级的个体数；如果把第二等级个体遍历完，那么该值为第一和二等级的个体总数（累加）

%% CSCD
    for front = 1 : (length(F) - 1) % 遍历所有前沿
        crowd_dist_obj = 0;
        y = [];
        previous_index = current_index + 1;  % 1
        % F中存储的是每一个rank中个体的索引
        % 获得到的y就是rank为front的子种群
        for i = 1 : length(F(front).f) % 遍历每个前沿中每个个体
            y(i,:) = sorted_based_on_front(current_index + i,:); % 把第一行的所有值赋值给y的第一行
        end
        current_index = current_index + i; 
        % S基于目标空间排序
        sorted_based_on_objective = [];
        for i = n_var+1 : n_obj+n_var % i=3:4
            [sorted_based_on_objective, index_of_objectives] = ...
                sort(y(:,i));   % 按目标值f1排序，把f1一列的值存到sorted_based_on_objective表中(26*1)，26是第一等级的个体数
            sorted_based_on_objective = []; %清零，通过后面遍历得到，为什么？
            % 获得基于某一个目标空间的排序
            % 和决策空间的算法设计基本一致，无k-means
            for j = 1 : length(index_of_objectives) % j=1:26，遍历第一等级所有个体
                sorted_based_on_objective(j,:) = y(index_of_objectives(j),:); % sorted_based_on_objective存放按f1从小到大排列的数
            end
            f_max = ...
                sorted_based_on_objective(length(index_of_objectives), i); %在按f1从小到大排列的表中，f1最大值为最后一个个体的f1
            f_min = sorted_based_on_objective(1,  i); 

            if length(index_of_objectives)==1 %如果第一等级（当前前沿）个体只有1个
                y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;  % y的9行（f1最小的个体）8列设为1
            elseif i>n_var
                % 计算目标空间的边缘个体的拥挤度
                % 在最小化问题中，设定到低边界点的最大距离和到上边界点的最小距离。
                y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;  % y的9行（当前等级f1最小的个体）8列设为1
                y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)=0; % y的26行（当前等级f1最大的个体）8列设为0
            end
             % 计算目标空间的中间个体的拥挤度
             for j = 2 : length(index_of_objectives) - 1
                next_obj  = sorted_based_on_objective(j + 1, i); %第3个个体的f1值，即后一个个体
                previous_obj  = sorted_based_on_objective(j - 1,i); %第1个个体的f1值，即前一个个体
                if (f_max - f_min == 0)
                    y(index_of_objectives(j),n_obj + n_var + 1 + i) = 1; % 放第9列
                else
                    y(index_of_objectives(j),n_obj + n_var + 1 + i) = ...
                         (next_obj - previous_obj)/(f_max - f_min);  % 第2个个体的拥挤度：(第3个个体的x1-第1个个体的x1)/(f_max - f_min)
                end
             end
        end
        % 截止到目前，已经计算出决策空间和目标空间的拥挤度距离
        %% 计算目标空间的拥挤度距离（求平均）
        crowd_dist_obj = [];
        % 每一个前沿计算目标空间拥挤度距离
        crowd_dist_obj(:,1) = zeros(length(F(front).f),1);
        for i = 1 : n_obj  %i=1:2
            crowd_dist_obj(:,1) = crowd_dist_obj(:,1) + y(:,n_obj + n_var + 1 + n_var + i); %把y的前8列（根据f1算出的拥挤度距离）放到 crowd_dist_obj表(24*1)中
        end
        crowd_dist_obj=crowd_dist_obj./n_obj; % 目标空间两个拥挤度距离相加求平均
        avg_crowd_dist_obj=mean(crowd_dist_obj); % 目标空间的拥挤度距离平均值
    %% 计算特殊的拥挤度距离
        special_crowd_dist=zeros(length(F(front).f),1); % 24*1，24是第一等级的个体数
        avg_crowd_dist_var = mean(y(:,n_var+n_obj+2)); % 决策空间的拥挤度距离平均值
        for i = 1 : length(F(front).f) % i=1:24
            crowd_dist_var(i) = y(i,n_var+n_obj+2); %y表的1行6列（决策空间的拥挤度距离）
            % 该个体的目标空间拥挤度值大于平均值或者决策空间拥挤度值大于平均值
            if crowd_dist_obj(i)>avg_crowd_dist_obj||crowd_dist_var(i)>avg_crowd_dist_var
                special_crowd_dist(i)=max(crowd_dist_obj(i),crowd_dist_var(i)); % 如果拥挤度距离大于平均值
            else
                special_crowd_dist(i)=min(crowd_dist_obj(i),crowd_dist_var(i)); % 
            end
        end
        y(:,n_obj + n_var + 3) = special_crowd_dist; % 第7列是 cscd值 特殊拥挤度距离
        y(:,n_obj+n_var+4)=crowd_dist_obj; % 第8列是 目标空间拥挤度距离
        [~,index_sorted_based_crowddist]=sort(special_crowd_dist,'descend');%按特殊拥挤度距离降序排序
        y=y(index_sorted_based_crowddist,:); % 第7列(特殊拥挤度距离)降序排列
        y = y(:,1 : n_obj + n_var+4 ); %取y的1-8列
        z(previous_index:current_index,:) = y; % 最后按照特殊拥挤度距离值降序排列，放到z表的1-24行
        % 开始遍历第二个前沿，y始终存放同一等级的个体
    end
f= z(); % f存放所有等级个体的所有算出的数据
end
