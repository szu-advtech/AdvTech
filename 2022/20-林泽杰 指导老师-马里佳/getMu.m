function Mu = getMu(data,Z,N,K,D)
%GETMU 此处显示有关此函数的摘要
%   此处显示详细说明
Mu = zeros(K,D);
% a=zeros(1,K);
for j=1:K
    a=zeros(1,D);
%     b=zeros(1,D);
    for i=1:N
        a = a + Z(i,j)*data(i,:);
    end
    b = sum(Z(:,j));
    Mu(j,:)=a/b;
end

% for j=1:K
%     b(j) = sum(Z(:,j));
% end


end

