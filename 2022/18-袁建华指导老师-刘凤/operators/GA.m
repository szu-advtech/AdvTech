function Offspring = GA(Parent,rmp,Pc,disC,sigma)
% 此函数功能是通过模拟二进制交叉和高斯变异产生子代，并利用垂直文化传播进行技能因子的继承（两两组成的父代个体，必须进行交叉或者变异）。
% Input: Parent父代信息（染色体，技能因子）、rmp文化交流参数、Pc模拟二进制交叉概率、disC交叉参数、sigma高斯变异参数
% Output: Offspring子代信息（染色体，技能因子）
% 第“1.”模式中，通过变异产生的两个后代可能具有相同的父母；第“2.”模式中，通过变异产生的两个后代父母一定不同
    [N,~] = size(Parent.rnvec);
    select = randperm(N);
    rrnvec = Parent.rnvec(select,:);%打乱顺序
    sskill_factor = Parent.skill_factor(select,:); 
    Parent1 = rrnvec(1:floor(end/2),:);
    factor1 = sskill_factor(1:floor(end/2),:);
    Parent2 = rrnvec(floor(end/2)+1:floor(end/2)*2,:); % floor:向负无穷舍入
    factor2 = sskill_factor(floor(end/2)+1:floor(end/2)*2,:);
    Offspring = INDIVIDUAL();
    Offspring.skill_factor = sskill_factor;%初始化子代的技能因子对应为父代的技能因子
    factorb1 = repmat(1:N/2,1,2);
    factorb2 = repmat(N/2+1:N,1,2);
    temp = randi(2,1,N);%对于子代随机选择它是继承第一个父母还是第二个父母
    offactor = zeros(1,N);
    offactor(temp == 1) = factorb1(temp == 1);
    offactor(temp == 2) = factorb2(temp == 2);%子代继承父母的编号
    [NN,D]   = size(Parent1);
    
    % Simulated binary crossover
    beta = zeros(NN,D);
    mu   = rand(NN,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta(repmat(factor1 ~= factor2 & rand(NN,1)>=rmp,1,D)) = 1;%不同技能因子的个体只有满足rmp才能交叉
    beta(repmat(rand(NN,1)>=Pc,1,D)) = 1;
    Offspring.rnvec = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
    Offspring.skill_factor(repmat(beta(:,1) ~= 1,2,1)) = sskill_factor(offactor(repmat(beta(:,1) ~= 1,2,1)));%对于交叉的个体，其随机选择一个父代继承基因。
    
    % mutation
    rvec=normrnd(0,sigma,[N,D]);
    Offspring.rnvec(repmat(beta(:,1) == 1,2,D)) = Offspring.rnvec(repmat(beta(:,1) == 1,2,D)) + rvec(repmat(beta(:,1) == 1,2,D));%只对没有交叉的个体进行变异
    %编码
    Offspring.rnvec(Offspring.rnvec>1)=1;
    Offspring.rnvec(Offspring.rnvec<0)=0;
end