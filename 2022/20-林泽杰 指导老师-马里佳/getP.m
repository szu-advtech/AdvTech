function p = getP(data,idx,Mu,Cov,N,D,K)
%GETP 求pk
%   data:数据集 idx:k-means方法的分类列表
%   Mu:均值mu（K=1,2,3） Cov1,2,3分别代表协方差矩阵(K=1,2,3)
%   N：数量点  D:数据集维度   K:聚类数
p = zeros(N,K);
Wine1=[];Wine2=[];Wine3=[];
for i=1:N
    if idx(i)==1
        Wine1 = [Wine1;[data(i,:),i]];
    elseif idx(i)==2
        Wine2 = [Wine2;[data(i,:),i]];
    elseif idx(i)==3
        Wine3 = [Wine3;[data(i,:),i]];
    end
end
% disp(Wine1);
[m,~]=size(Wine1);
p1 = mvnpdf(Wine1(:,1:D),Mu(1,:),Cov(:,:,1));
for i=1:m
    p(Wine1(i,D+1),1)=p1(i);
end

[m,~]=size(Wine2);
p2 = mvnpdf(Wine2(:,1:D),Mu(2,:),Cov(:,:,2));
for i=1:m
    p(Wine2(i,D+1),2)=p2(i);
end

[m,~]=size(Wine3);
p3 = mvnpdf(Wine3(:,1:D),Mu(3,:),Cov(:,:,3));
for i=1:m
    p(Wine3(i,D+1),3)=p3(i);
end


end

