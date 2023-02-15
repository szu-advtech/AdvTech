% func_name fname:'MMF1' and so on
% VRmin xl:[1,-1]
% VRmax xu:[3,1]
% n_obj n_obj:2
% NP popsize:200
% Max_Gen Max_Gen:100
function [ps,pf] = MMO_DE_CSCD( func_name,VRmin,VRmax,n_obj,NP,Max_Gen )
ps = cell(Max_Gen,1);
pf = cell(Max_Gen,1);
n_var=size(VRmin,2);    %取决策空间的维度，取VRmin的列数
Max_FES=Max_Gen*NP;     %最大适应度评估
%% 初始化种群
VRmin=repmat(VRmin,NP,1); %% repmat 快速产生一个较大的矩阵，VRmin原本是决策变量的下限[1,-1]（1行2列），最后变成200行2列
VRmax=repmat(VRmax,NP,1);
% rand 产生随机变量的形状为 (200, 2) 相当于200个个体，每个个体由两个变量组成
pos=VRmin+(VRmax-VRmin).*rand(NP,n_var); %initialize the positions of the individuals pos是初始种群
%% 评价种群，计算每个个体的适应度f值
fitness=zeros(NP,n_obj);   % 适应度列数跟目标函数个数有关
for i=1:NP   %1:200
    fitness(i,:)=feval(func_name,pos(i,:));  % feval函数执行指定的函数，这里是把值带入到MMF1函数中，得到f1 f2（适应度），一行一行（一个个体一个个体）地变化
end
fitcount=NP; % 计算适应度评估，为种群大小，即200，一次适应度评估需要200个个体作为一批进行评估（评估分为20000/200=100批）
G=0; %初始化迭代次数

%% 主循环
while fitcount <= Max_FES   % 如果当前适应度评估小于最大适应度评估，200<20000或400<20000
    G=G+1;
    Parentstar=[pos,fitness];  % pos是200×2，第一个父代，每个个体的两个x（决策变量）和适应度两个y（目标值）拼接，200×4
    K=10;
    Parentstar = non_domination_scd_kmeans_sort(Parentstar,n_obj,n_var,K); % Parentstar是按特殊拥挤度距离值降序排列的数据，n_var+n_obj+1: the level of front前沿等级   +2：the decision space distance决策空间距离 +3：CSCD value基于聚类的特殊拥挤距离  +4：the objective space distance目标空间距离
    
    %% 计算每个前沿的个体数
    Total_pf = max(Parentstar(:,n_var+n_obj+1));  % 前沿的个数，这里是21（等级1-21）
    pop = cell(Total_pf,1);  %pop是21*1，每空都是一个空矩阵，每个矩阵是不同等级的所有个体
    num = zeros(Total_pf,1);  %num是21*1
    
    for i=1:NP % i=1:200
        pop{Parentstar(i,n_obj+n_var+1)} = [ pop{Parentstar(i,n_obj+n_var+1)};Parentstar(i,:)]; %按照等级存放所有数据到pop里，将等级1的所有个体存放在pop第1行1列的矩阵里，将等级2的所有个体存放在pop第2行1列空里
        num(Parentstar(i,n_obj+n_var+1)) = num(Parentstar(i,n_obj+n_var+1)) + 1; %初始num(Parentstar(i,n_obj+n_var+1))是num(1)=0，num(21*1)存放每个等级的个体数
    end
    
    % 根据 CSCD 值对每个前沿中的个体进行排序
    for i=1:Total_pf % i=1:21
        [~,index_sorted_on_scd]= sort(pop{i,1}(:,n_var+n_obj+3),'descend');  %pop{1,1}是pop里1行1列（等级1）的所有个体，按照第7列（cscd值 特殊拥挤度距离计算）降序排序，返回对应索引
        pop{i,1}= pop{i,1}(index_sorted_on_scd,:);  % 根据索引对pop中的数组进行排序，返回按第7列（cscd值 特殊拥挤度距离计算）降序排序的结果
    end
    %% DBESM 
    offspring = [];
    for i=1: Total_pf
        u = zeros(num(i),n_var); %num(1)是等级1的个体数
        for j= 1 : size(pop{i,1},1) %size(pop{1,1},1)是等级1的个体数
            % 根据所提出的DBESM机制选择样本
            if i==1  %当等级为1时
                r_pf(1:3) = 1; %r_pf的1行3列都为1（3*1），即[1 1 1]
            else     %当等级大于1时
                r_pf = ceil(rand(1,3)*(i-1)); %ceil是向上取整，rand(1,3)生成0-1之间的三个数，i=2时矩阵全乘以(2-1)，即[0 0 0]
            end
            
            if  num(r_pf(1)) < 3 % 如果等级1的个体数小于3，i=1，r_pf=[1 1 1]，r_pf(1)是[1 1 1]中的第一个数，num(1)是等级1的个体数，即21
                r1 = ceil(rand* num(r_pf(1)));  %rand乘以等级1的个体数，再向上取整
            else  % 如果等级1的个体数大于等于3
                r1 = min(ceil(0.1 * num(r_pf(1))),3);  % p=0.1;把等级1的个体数缩小10倍，然后向上取整，取它和3的最小值。一般r1=3
            end

            % 计算距离，即与周围的三个点在决策空间上的距离
            % [1.79 1.79 1.79;0.67 0.67 0.67](3*2)-[1.79 1.74 1.22;0.67 0.93 -0.79]=[0 0.0508 0.5619;0 -0.2539 1.4701]，两列各自平方，两列相加（3*1）
            distance = sum((repmat(pop{i,1}(j,1:n_var), r1 ,1) - pop{r_pf(1),1}(1:r1,1:n_var)).^2,2); %将等级1个体的第1个个体前两列（决策变量）复制成r1（3）行2列，pop1行1列（等级1的所有个体）的前3行前2列（前三个个体的决策变量），相减（3*2 第一行全为0），每列各自平方，两列相加（3*1）
            if length(distance) == 1 % length(distance)是3
                select_index = r1;
            else
                if length(find(max(distance)==distance))== r1 %find找distance的最大值在distance里的索引，length看有多少个相等的最大值，如果等于3
                    select_index = ceil(rand*r1); %select_index是随机产生，3倍的rand（就一个数），向上取整
                else
                    distance = max(distance) - distance; %0.6865-[0 0.6865 0.1510](3*1)=[0.6865 0 0.5355]
                    select_pro =  distance./(sum(distance)); %[0.6865 0 0.5355]./(0.6865+0+0.5355)=[0.5618 0 0.4382]
                    true_pro = cumsum(select_pro); %累积和，[0.5618 0.5618 1]，第二个是1 2相加，第三个等于1 2 3相加
                    select_index = find(rand<true_pro); %找随机数小于true_pro的个体索引，这里是2 3
                end
            end
            
            r2 = ceil(rand*num(r_pf(2)));  %r_pf(2)=1，num(1)是等级1的个体数，乘以随机数，向上取整
            r3 = ceil(rand*num(r_pf(3)));

            %% 变异
            % v是1*2，等级1的第1个个体前2列+0.8*（等级1的select_index(1)=2行个体前2列-等级1的1行个体前2列+等级1的r2(22)行个体前2列-等级1的r3(2)行个体前2列）
            v = pop{i,1}(j,1:n_var) + 0.8*(pop{r_pf(1),1}(select_index(1),1:n_var) - pop{i,1}(j,1:n_var) + pop{r_pf(2),1}(r2,1:n_var) -pop{r_pf(3),1}(r3,1:n_var));
            v = boundConstraint(v, pop{i,1}(j,:), [VRmin(1,:);VRmax(1,:)]); %越界的数变成不越界的数
            %% 交叉
            jrand = ceil(rand*n_var); % 2或1
            for h=1:n_var %h=1:2
                if jrand<1 || h==jrand
                    u(j,h) = v(1,h); % 把变异后的v[1.036 -0.98]的第一个数或第二个数给u
                else
                    u(j,h) = pop{i,1}(j,h); % u（28*2）放等级1个体1的第1列x1或第2列x2
                end
            end
        end
        offspring = [offspring;u]; % 产生子代，u表数据加在offspring(2列)表下面，行数为前i个等级个体数之和，最后肯定有200行
    end
    
    % 评价子代
    for i=1:size(offspring,1) % i=1:200
        offspring_fitness(i,:)=feval(func_name,offspring(i,:)); %一次两个决策变量x1x2按MMF1来生成两个适应度f，存放在offspring_fitness表中
        fitcount = fitcount +1 ;  % 原始fitcount=200，一次计算评价次数加一
    end
    temp_pop = [pos,fitness;offspring,offspring_fitness]; %400*4，存放初始种群(200*2)、适应度(200*2)、子代种群(另起一行)、适应度
    temp_pop = non_domination_scd_kmeans_sort(temp_pop ,n_obj,n_var,K); %400*8，子代也经过非支配kmeans排序
    
    % Environmental seceltion
    Parent = replace_decision_chromosome_kmeans(temp_pop,n_obj,n_var, NP);
    pos = Parent(:,1:n_var); % pos是200*2
    fitness = Parent(:,n_var+1:n_var+n_obj); % Parent表的3-4列放f1和f2
    ps_gen = Parent(Parent(:,n_var+n_obj+1)==1,1:n_var); 
    ps{G,1} = Parent(Parent(:,n_var+n_obj+1)==1,1:n_var); % 取出Parent表里第5列（等级）等于1的行，再取前2列，等级为1的才叫前沿啊！
    pf{G,1} = Parent(Parent(:,n_var+n_obj+1)==1,n_var+1:n_var+n_obj);
end


