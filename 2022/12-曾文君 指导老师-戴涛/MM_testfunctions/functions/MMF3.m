function f = MMF3(x)
% 0<=x1<= 1  0<=x2<=1.5
   f=zeros(2,1);
    f(1)      = x(1);
    if x(2)>=0&&x(2)<=0.5                       %0<=x(2)<=0.5
        y2=x(2)-x(1)^0.5;
    end
    if x(2)>0.5&&x(2)<1&&x(1)>=0&&x(1)<=0.25    %0.5<x(2)<1&&0<=x(1)<=0.25
        y2=x(2)-0.5-x(1)^0.5;
    end
    if x(2)>0.5&&x(2)<1&&x(1)>0.25              %0.5<x(2)<1&&x(1)>0.25
        y2=x(2)-x(1)^0.5;
    end
    if x(2)>=1&&x(2)<=1.5                             %1<=x(2)<=1.5           
        y2=x(2)-0.5-x(1)^0.5;
    end
   f(2)      = 1.0 - (x(1))^0.5 + 2*((4*y2^2)-2*cos(20*y2*pi/sqrt(2))+2);
end

% %% generate ture PS and true PF
% no=400;                  %帕累托最优解集中解的个数
% PS(1:no/2,1)=linspace(0,1,no/2);
% PS(1:no/2,2)=sqrt(PS(1:no/2,1));
% PS(no/2+1:no,1)=linspace(0,1,no/2);
% PS(no/2+1:no,2)=PS(1:no/2,2)+0.5;
% 
% PF(:,1)     = linspace(0,1,no/2);
% PF(:,2)     = 1-sqrt(PF(:,1));
% figure(1)
% plot(PS(:,1),PS(:,2),'ro');
% figure(2)
% plot(PF(:,1),PF(:,2),'ro');


