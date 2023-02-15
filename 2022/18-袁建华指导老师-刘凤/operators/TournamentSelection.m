function index = TournamentSelection(K,N,Fitness)
% TournamentSelection进制锦标赛(适应度值越大，被选概率越高)
% Input: K几元锦标赛，N选出N个个体，Fitness适应度向量
% Output：index选出个体的索引
    Fitness = reshape(Fitness,1,[]);
    if (~isempty(find(Fitness<1, 1)))
        Fitness(Fitness == 0) = inf;
        Fitness = 1/min(Fitness)*Fitness;%归一化
    end
    [~,rank] = sort(Fitness,'descend');
    [~,rank] = sort(rank);
    Parents  = randi(length(Fitness),K,N);
    [~,best] = min(rank(Parents),[],1);
    index    = Parents(best+(0:N-1)*K);
end