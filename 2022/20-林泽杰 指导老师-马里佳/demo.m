A = readmatrix('./archive/Wine.csv');
[m,n] = size(A);
data = A(:,2:n);
[N,D] = size(data);     % 获取数据集的N和D（N=m,D=n-1）
K=3;
[idx,Mu] = kmeans(data,K);   % idx是类别，C是聚类质心位置

% E
Cov = zeros(D,D,K);
for i=1:K
    Cov(:,:,i) = cov(data);        % 初始化GMM的协方差矩阵\Sigma
end
pi = getPi(idx,N,K);    % 求权重\pi   从K=1到K=3
p = getP(data,idx,Mu,Cov,N,D,K);   % 求pk  ——高斯分布的概率密度函数
Z = getZ(pi,p,N,K);     % 求Z(i,k)

% M
for i=1:K
    pi(i) = sum(Z(:,i))/N;  % 新的\pi
end

Mu = getMu(data,Z,N,K,D);  % 新的\mu
Min(data,Z,Mu,Cov,1,N,D);

function min=Min(data,Z,Mu,Cov,k,N,D)
% U=zeros(D);
% D=zeros(D);
% inverse_C=zeros(D);

[U,V]=eig(Cov(:,:,k));   % U:Q（列向量是特征向量） V:新坐标系的协方差矩阵
inverse_C=inv(V);   % 新坐标系的逆协方差矩阵~C
Det_inverse_C = det(inverse_C);          % ~C的行列式
fprintf('Det_inverse_C=%f\n',Det_inverse_C);
fprintf('tr(Det_inverse_C)=%f\n',trace(inverse_C));
disp(inverse_C);
% tr=trace();         % 求矩阵的迹
% StS = zeros(D);
s=zeros(D);
for i=1:N
    s = s+Z(i,k)/sum(Z(:,k))*(data(i,:)-Mu(k,:))*(data(i,:)-Mu(k,:)).';
end
StS=N*U*s*U.';
tr = trace(inverse_C*StS);   % 求迹
min = -log(Det_inverse_C)+tr/N;
fprintf('min=%f\n',min);
disp(min);
% Ck(k)=argmin(-log(Det_inverse_C)+tr);
end