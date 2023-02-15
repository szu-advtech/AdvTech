function [ R ] = RBFcirle( Xp )
%RBFCIRLE Summary of this function goes here
%   确定最大距离
k=length(Xp(1,:));
p=length(Xp(:,1));
for i=1:p
    X=Xp;
    X(i,:)=[];
    for j=1:p-1
        A(j)=norm((Xp(i,:)-X(j,:)));
    end
    R(i)=max(A);
    R(i)=R(i)/((k^0.5)*(p-1)^(1/k));
end
end

