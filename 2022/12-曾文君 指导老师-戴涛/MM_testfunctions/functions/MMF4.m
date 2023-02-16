function f = MMF4(x)
% -1<=x1<= 1  0<=x2<=2
   f=zeros(2,1);
    if x(2)>1
        x(2)=x(2)-1;
    end
    f(1)      = abs(x(1));  
    f(2)      = 1.0 - (x(1))^2 + 2*(x(2)-sin(pi*abs(x(1))))^2;

end