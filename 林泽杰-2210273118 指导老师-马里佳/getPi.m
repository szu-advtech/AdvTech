function pi = getPi(idx,N,K)
%GETPI 求每一类的权重\pi
%   idx:k-means方法的分类列表 N：数量点 K:聚类数
count=zeros(1,K);
for i=1:N
    j = idx(i);
    count(j)=count(j)+1;
end
pi=count/N;
end

