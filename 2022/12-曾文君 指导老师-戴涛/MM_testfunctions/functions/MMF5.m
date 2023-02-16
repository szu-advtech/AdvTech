function Obj = MMF5(x)
% 1<=x1<=3    -1<=x2<=3
   
   Obj = zeros(2,1);
    if x(2)>1
        x(2)=x(2)-2;
    end
    Obj(1)      = abs(x(1)-2);             
    Obj(2)     = 1.0 - sqrt( abs(x(1)-2)) + 2.0*( x(2)-sin(6*pi* abs(x(1)-2)+pi)).^2;
end