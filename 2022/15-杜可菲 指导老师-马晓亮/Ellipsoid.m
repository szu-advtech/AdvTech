 function [ T ] = Ellipsoid( X )
%ELLIPSOID50 Summary of this function goes here
Dim = size(X,2);
T=0;
for i=1:Dim
    T=T+i*X(:,i).^2;
end
end

