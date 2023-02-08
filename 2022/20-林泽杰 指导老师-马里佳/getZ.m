function Z = getZ(pi,p,N,K)
%GETZ 此处显示有关此函数的摘要
%   pi:权重  p:pk概率  N：数量点  K:聚类数
Z = zeros(N,K);
for i=1:N
    s = pi*p(i,:)';
    for j=1:K
        Z(i,j) = pi(j)*p(i,j)/s;
%         s = 0;
%         for t=1:K
%             s = s + pi(t)*p(i,t);
%         end
%         pi*p(i,:)';
    end
end

end

