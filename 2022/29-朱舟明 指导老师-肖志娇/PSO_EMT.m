%获得个体最优、全局最优
function [result] = PSO_EMT(data, label, dataName, fold,promisingFeature)

    % 超参数设置
    popSize = 100;%种群大小
    Iterations = 100;%最大迭代次数
    featureNum = size(data,2);%data的列数量
    upperBound = 0.7;%用于缩小搜索空间
    scaleFactor = 0.05;%子集更新机制中的比例因子
    subUpdateIter = 10;%子集更新机制中的最大迭代次数
    RMP = 0.6;
    c1 = 1.49445;%学习因子
    c2 = 1.49445;
    c3 = 1.49445;

    Vlimit = [-0.6,0.6];

    % 继承相关任务
    KN_point = promisingFeature.KN_point;
    weights = promisingFeature.weights;
    subset = promisingFeature.subset;

    % 初始化种群
    [position,maxValue] = initPop(popSize,featureNum,weights,upperBound,KN_point);
    V = (rand(size(position)) - 0.5) * Vlimit(2)*2;

    [skillFactor,~,fit1,fit2] = getSkillFactorFit(data,label,position,subset);%得到初始的适应度和技能因子


    % 初始化pbest和gbest
    pBest.task1.pos = position; 
    pBest.task2.pos = position; 
    pBest.task1.fit = fit1; 
    pBest.task2.fit = fit2;
    pBest.task1.mask = repmat(subset,popSize,1);

    [~,index1] = min(fit1);
    [~,index2] = min(fit2);
    gBest.task1.pos = position(index1,:); 
    gBest.task2.pos = position(index2,:); 
    gBest.task1.fit = fit1(index1); 
    gBest.task2.fit = fit2(index2);
    gBest.task1.mask = subset;

    updateSubsetFlag = 0;
    for iter = 1:Iterations
        if mod(iter,10)==1
            disp(strcat("PSO_EMT on, ", dataName," Iter ", num2str(iter),' ,best Fitness ==',num2str(gBest.task1.fit),' ,time==',num2str(toc)));
        end
        w = 0.9-0.5*iter/Iterations;%惯性因子

        V = assortativeMating(V,position,skillFactor,pBest,gBest,c1,c2,c3,w,RMP);
        V(V<Vlimit(1)) = Vlimit(1);
        V(V>Vlimit(2)) = Vlimit(2);

        position = position + V;
        %位置越界处理：
        position(position<0) = 0;
        position(position>maxValue) = maxValue(position>maxValue);

        [fitness,skillFactor] = getFitness(data,label,position,subset,skillFactor);
        pBest = updatePBest(pBest,position,fitness,subset,skillFactor);
        [gBest,flag] = updateGBest(gBest,position,fitness,subset,skillFactor);

        if ~flag
            updateSubsetFlag = 1 + updateSubsetFlag;
        else
            updateSubsetFlag = 0;
        end

        if (updateSubsetFlag==subUpdateIter) && (iter~=Iterations)
            disp('subset Update!')
            updateSubsetFlag = 0;
            [subset] = subsetUpdating(subset,scaleFactor);
        end
    end
    
    result.fold = fold;
    result.name = dataName;
    result.pbest = pBest;
    result.gbest = gBest;

end

