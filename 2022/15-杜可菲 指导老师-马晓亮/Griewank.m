function [ T ] = Griewank( X )
%GRIEWANK30 Summary of this function goes here
%   Detailed explanation goes here
Dim = size(X,2);
T=0;
S=1;
for i=1:Dim
    T=T+(X(:,i)).^2;
    S=S.*cos(X(:,i)./(i^0.5));
end
T=1/4000.*T-S+1;

end

