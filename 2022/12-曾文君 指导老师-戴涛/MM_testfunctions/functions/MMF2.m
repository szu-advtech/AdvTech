function f = MMF2(x)
% 0<=x1<= 1  0<=x2<=2
   f=zeros(2,1);
    if x(2)>1
        x(2)=x(2)-1;
    end
    f(1)      = x(1);  
    y2=x(2)-x(1)^0.5;
    f(2)      = 1.0 - sqrt(x(1)) + 2*((4*y2^2)-2*cos(20*y2*pi/sqrt(2))+2);
 
end