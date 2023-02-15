function [ T ] = Rosenbrock( X )
%ROSENBROCK20 Summary of this function goes here
% %%%%%ShifteDim Rosenbrock¡¯s Function [-100 100] F6
Dim = size(X,2);
T=0;
for i=1:Dim-1
    T=T+100.*(X(:,i+1)-X(:,i).^2).^2+(1-X(:,i)).^2;
end  

end

