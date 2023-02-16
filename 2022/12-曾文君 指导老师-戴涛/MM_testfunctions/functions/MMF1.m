function y = MMF1(x)
% 1<=x1<=3    -1<=x2<=1
    y=zeros(2,1);
    x(1)=abs((x(1)-2));
    y(1)      = x(1);             
    y(2)      = 1.0 - sqrt(x(1)) + 2.0*(x(2)-sin(6*pi*x(1)+pi))^2;
 
end