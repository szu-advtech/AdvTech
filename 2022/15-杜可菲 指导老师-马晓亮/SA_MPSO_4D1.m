format compact;
clear all;

clearvars
t1=clock;
%% 初始化
warning('off')
%     iterl = 50;
iterl = 200;

%% The dimension of the problem 问题的维度
c = 50;
% c = 30;
% c = 10;
nl = 5*c;
npl = 100;
np = 50;

%% Set the maximum number of function evalutions  % 设置函数评估值的最大值
if c >= 30
% if c >= 10
%     N = 100;    %种群中个体的个数
    N = 4*c-1;
    maxFES = 500;
%     termination = 500;
else
    N = 4*c-1;
    maxFES = 11*c;
    
end

%% The number of total running times 实际运行次数
totaltime=5;

%% Record the historical best result of all problems 记录所有问题的历史最优结果
Fbest=[];

for problem = 1:5
    %% 设置每个问题的下界和上界
    if problem==1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%_____1.单峰函数Ellipsoid
        name=@Ellipsoid;
        global bu bd
        bu=5.12;
        bd=-5.12;
        vmax=5.12;
        vmin=-5.12;
    end
    if problem==2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%_____2.多峰函数Rosenbrock
        name=@Rosenbrock;
        global bu bd
        bu=2.048;
        bd=-2.048;
        vmax=2.048;
        vmin=-2.048;
    end
    if problem==3
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%_____3.多峰函数Griewank
        name=@Griewank;
        global bu bd
        bu=600;
        bd=-600;
        vmax=600;
        vmin=-600;
    end
    if problem==4
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%_____4.多峰函数Ackley
        name=@Ackley;
        global  bu bd
        bu=32.768;
        bd=-32.768;
        vmax=32.768;
        vmin=-32.768;
    end
    if problem==5
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%_____4.多峰函数Ackley
        name=@Rastrigin;
        global  bu bd
        bu=5;
        bd=-5;
        vmax=5;
        vmin=-5;
    end
    
    %% Record the historical best result for each problem  对于每个问题记录历史最佳结果
    fbest=[];
    
    %% time表示运行的次数
    for time=1:totaltime %运行totaltime次
        BU = repmat(bu,1,c);BD = repmat(bd,1,c);
        lu = [bd* ones(1, c); bu * ones(1, c)];
        [ POP ] = initialize_pop(N,c,bu,bd);    %初始化种群：N种群中个体的个数，c为维度
        obj = feval(name,POP);  %真实函数评估
        n = [N];
        bound = [bd,bu];
        X = POP;
        slst = AP(X);
        citr = zeros(size(slst,2), 1);
        Pbestx = zeros(n(1), c);
        Pbestf = zeros(n(1), 1);
        Gbestx = zeros(size(slst,2), c);
        Gbestf = zeros(size(slst,2), 1);
        f = zeros(n(1), 1);
        v = zeros(n(1), c);
        for j = 1 : size(slst,2)
            h = [];
            for i = 1:size(slst{j},2)
                h(i) = obj(slst{j}(i),:);
                f(slst{j}(i)) = h(i);
                Pbestx(slst{j}(i),:) = X(slst{j}(i),:);
                Pbestf(slst{j}(i)) = h(i);
                v(slst{j}(i),:) = bu - (bu - (bd))*rand(1, c);
            end
            Gbestf(j) = min(h);
            Gbestx(j,:) = X(slst{j}(find(h == min(h))),:);
        end
        Roverlap = 0.6;
        Rexcl = 0.5*(1/size(slst,2))*(bu - bd)/(size(slst,2)^(1/c)); %Rexcl为置信半径△k
        condition = zeros(size(slst,2),1);
        Data = [POP,obj];   %个体坐标值+真实评估值
        Record=[];
        mop.domain = [BD;BU];
        StepNext = 1;
        a0 = 0;
        FES = 0;
        while size(Data,1)< maxFES
%         while FES < maxFES
            StepCur = StepNext;
            switch StepCur
                case 1
                  %% 全局搜索
                    %                 'Start Global Serch'
                    [Exactly,condition,citr,slst,X, f, v, Pbestx, Pbestf,b,state] = Global_search(Data,bu,bd,condition,slst,n,maxFES,citr,X, f, v, Pbestx, Pbestf, Gbestx, Gbestf);
                    %                 [Exactly,condition,citr,slst,X, f, v, Pbestx, Pbestf,b,state] = Global_search1(Data,bu,bd,condition,slst,n,termination,citr,X, f, v, Pbestx, Pbestf, Gbestx, Gbestf);
                    %                 [Exactly,condition,citr,slst,X, f, v, Pbestx, Pbestf,b,state] = Global_search2(Data,bu,bd,condition,slst,n,termination,citr,X, f, v, Pbestx, Pbestf, Gbestx, Gbestf);
                    
                case 2
                    %%% 选择的合适的区域进行局部搜寻
                    Pl = sortrows(Data,c+1);
                    meanX = Pl(1,:);
                    bul = meanX(:,1:c) + Rexcl;
                    bul(bul > bu) = bu;
                    bdl = meanX(:,1:c) - Rexcl;
                    bdl(bdl < bd) = bd;
                  %% 选取合适的点建立局部模型
                    Pl = unique(Pl,'rows');
                    D=Pl;
                    a111 = [];
                    for i = 1:size(D,1)
                        distance = sum(abs(Pl(:,1:c)-repmat(Pl(i,1:c),size(Pl,1),1)).^2,2).^(1/2);
                        a1111=find(distance < 0.001);
                        a1111(find(a1111==i)) = [];
                        a111=[a111;a1111];
                    end
                    Pl(a111,:) = [];
                    distance = sum(abs(Pl(:,1:c)-repmat(meanX(:,1:c),size(Pl,1),1)).^2,2).^(1/2);
                    [a1, b]= sortrows(distance,1);%对距离进行排序，其中a为距离，b为对应的下标
                    a11 = find(a1<Rexcl);
                    if size(Pl,1) < nl %+ 0.1*nl
                        Datal = Pl;
                    else if length(a11) < nl %+ 0.1*nl
                            Datal = Pl(b(1:nl),:);
                            
                        else
                            Datal = Pl(a11,:);
                        end
                    end
                  %% 建立局部模型对其进行局部搜索
                    %                 'Start Local Serch'
                    [gbestxf, gbestf, gbest] = CSO_LRA(Datal,bul,bdl,npl,iterl,meanX, nl,meanX);
                    %                 [gbestxf, gbestf, gbest] = CSO_LRA1(Datal,bul,bdl,npl,iterl,meanX, nl,meanX);
                    %                 [gbestxf, gbestf, gbest] = CSO_LRA2(Datal,bul,bdl,npl,iterl,meanX, nl,meanX);
                    
            end
            switch StepCur
                case 1
                    minData = min(Data(:,c+1));
                    if ~isempty(Exactly)%真实函数评估
                        Exactly(:,c+1) = feval(name,Exactly(:,1:c));
%                       %% 记录函数评估值的数量
%                         FES = FES+1;
                        
                        Data = [Data;Exactly(:,1:c+1)];
                        best = Exactly;
                        bestx = Gbestx; bestf = Gbestf;
                        
%                       %% Record the best result 记录最好的结果
%                         fbest(FES,time)=min(best(:,c+1));
                        
                        bestx(find(state == 1),:) = best(:,1:c);
                        bestf(find(state == 1),:) = best(:,c+1);
                        
                        Gbestx((bestf < Gbestf),:) = bestx((bestf < Gbestf),:);
                        Gbestf((bestf < Gbestf),:) = bestf((bestf < Gbestf),:);
                        [condition, slst, Gbestx, Gbestf] = condition_detectation(X, condition, slst, Roverlap, Rexcl, Gbestx, Gbestf, Pbestf);
                        
                        indexc = find(condition==2);
                        for i = 1 : size(indexc,1)
                            DataD = Data;
                            indexD = randperm(size(DataD,1),size(slst{indexc(i)},2));
                            Dr = DataD(indexD',:);
                            X(slst{indexc(i)},:) = Dr(:,1:c);
                            f(slst{indexc(i)},:) = Dr(:,c+1);
                            Pbestx(slst{indexc(i)},:) = Dr(:,1:c);
                            Pbestf(slst{indexc(i)},:) = Dr(:,c+1);
                            Gbestf(indexc(i),:) = min(Dr(:,c+1));
                            [~, a2] = min(Dr(:,c+1));
                            Gbestx(indexc(i),:) = Dr(a2,1:c);
                            DataD(indexD',:) = [];Dr = [];
                            for j = 1:size(slst{indexc(i)},2)
                                v(slst{indexc(i)}(j),:) = bu - (bu - (bd))*rand(1, c);
                            end
                            condition(indexc(i)) = 0;
                        end
                    else
                        [~,indexf] = min(f);
                        Exactly = X(indexf,:);
                        if problem == 0 %使用其他测试函数
                            Exactly(:,c+1) = compute_objectives(Exactly(:,1:c),c,problem_name);
                        else
                            Exactly(:,c+1) = feval(name,Exactly(:,1:c));
%                          %% 记录函数评估值的数量
%                             FES = FES+1;
                            
                        end
                        Data = [Data;Exactly(:,1:c+1)];
                    end
                    
%                   %% Record the best result 记录最好的结果
%                     fbest(FES,time)=min(Exactly(:,c+1));
                    
                    if min(Data(:,c+1)) < minData
                        StepNext =  1;
                    else
                        StepNext =  2;   % swith local
                        D1 = sortrows(Data,c+1);
                        if length(D1) < 5*c
                            D1 = D1;
                        else
                            D1 = D1(1:5*c,:);
                        end
                        [~,xmin]=min(D1(:,c+1));
                        Xmin=D1(xmin,:);
                        Xmin = Xmin(:,1:c);
                        [~,xmax]=max(D1(:,c+1));
                        Xmax=D1(xmax,:);
                        Xmax = Xmax(:,1:c);
                        Rexcl=norm(Xmin-Xmax)/2;
                    end
                    
                case 2
                    if ~isempty(gbest)
                        gbest(:,c+1) = feval(name,gbest(:,1:c));
%                       %% 记录函数评估值的数量
%                         FES = FES+1;
                        
                        a = find(condition==0);
                        gbesta = Gbestx(a,:); gbestaf = Gbestf(a,:);
                        D1 = repmat(gbest(:,1:c),size(a,1),1);
                        distance1 = sum(((D1 - gbesta).^2),2).^0.5;
                        [~,b1] = min(distance1);
                        if gbest(:,c+1) < gbestaf(b1,:)
                            Gbestx(a(b1),:) = gbest(:,1:c);
                            Gbestf(a(b1),:) = gbest(:,c+1);
                        end
                        t = gbest;
                        Record = [Record;t];
                        if gbest(:,c+1) < min(Data(:,c+1))
                            StepNext =  2;
                            ro=(meanX(:,c+1)-gbest(:,c+1))/(gbestxf - gbestf-10^(-20));
                            if norm(meanX(:,1:c)-gbest(:,1:c)) < Rexcl
                                es=1;
                            end
                          %% 更新置信半径△k+1（Rexcl）
                            if norm(meanX(:,1:c)-gbest(:,1:c)) >= Rexcl
                                es=2;
                            end
                            if ro<0.25
                                Rexcl=0.25*Rexcl;
                            end
                            if ro>0.75
                                Rexcl=es*Rexcl;
                            end
                            if ro>=0.25&&ro<=0.75
                                Rexcl=Rexcl;
                            end
                        else
                            StepNext =  1;
                        end                      
                        Data=[Data;gbest(:,1:c+1)];
                    else
                        StepNext =  2;
                    end
                    a0 = a0 + 1;
                    index0 = find(condition==0);
                    if a0 < 1 %size(slst,2)
                        StepNext=2;
                    else
                        a0 = 0;
                        StepNext = StepNext; %switch global
                    end
            end
            StepPre=StepCur;
            
        end
%         fprintf('最优解为：')
%         min(Data(:,c+1))
%         size(Data,1);
          fbest(time) = min(Data(:,c+1));
    end
    
    %% 记录5个问题所有历史结果
    Fbest{problem} = fbest;
end

% save('D:\Program Files (x86)\Matlab\R2019b\bin\SA_MPSO\Fbest.mat','Fbest')

for problem=1:5 
    mean(Fbest{1,problem})
    std(Fbest{1,problem})
end
