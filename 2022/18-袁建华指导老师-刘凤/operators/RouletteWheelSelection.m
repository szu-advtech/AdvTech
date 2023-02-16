function index = RouletteWheelSelection(N,Fitness)
% RouletteWheelSelection轮盘度选择（适应度值越大，被选择概率越大）
% Input:N选出N个个体，Fitness适应度向量
% Output:index选出的下标
    Fitness = reshape(Fitness,1,[]);
    if (~isempty(find(Fitness<1, 1)))
        Fitness(Fitness == 0) = inf;
        Fitness = 1/min(Fitness)*Fitness;%归一化
    end
    Fitness = cumsum(Fitness);
    Fitness = Fitness./max(Fitness);
    index   = arrayfun(@(S)find(rand<=Fitness,1),1:N);%找到rand<=Fitness中最先出现的1个不为零的下标
end