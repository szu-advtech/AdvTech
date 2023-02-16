function Obj = MMF7(Var)
% 1<=x1<=3    -1<=x2<=1
   
   Obj = zeros(2,1);
   Var(1)= abs(Var(1)-2);  
   Obj(1)     = Var(1);             
   Obj(2)      = 1.0 - sqrt( Var(1)) +(Var(2)-(0.3*(Var(1).^2)*cos(24*pi*Var(1)+4*pi)+0.6*Var(1))*sin(6*pi*Var(1)+pi)).^2;
end