function Obj = MMF8(Var)
% -pi<x1<pi    0<=x2<=9
   
   Obj = zeros(2,1);
   if Var(2)>4
        Var(2)=Var(2)-4;
   end
  Obj(1)      = sin(abs(Var(1)));             
  Obj(2)    = sqrt(1.0 - (sin(abs(Var(1)))).^2) + 2.0*(Var(2)-(sin(abs(Var(1)))+abs(Var(1)))).^2;
end