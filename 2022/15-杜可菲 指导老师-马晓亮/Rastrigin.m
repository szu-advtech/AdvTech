function [ T ] = Rastrigin( X )

Dim = size(X,2);
T=0;
for i=1:Dim
    T=T+(X(:,i).^2-10*cos(2*pi*X(:,i))+10);
end
end