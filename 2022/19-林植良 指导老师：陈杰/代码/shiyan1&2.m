clear;
clc;
format short;
f=sym('[abs(x1)-x2*x3+(x2+x4)^2]')

g(1)=sym('[abs(2*x1-x3)-6]')
g(2)=sym('[2*x2-3*x1-5]')
g(3)=sym('[x1+x2+x3+x4-5]')

h1=sym('[x1+2*x2-x3-3]')

A=[1,2,-1,0]
I=[1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1]
P=A'*inv((A*A'))*A
 a0=0.00001;l=0.0002;

x10=1;
x20=1;
x30=0;
x40=1;


x1=0
x2=3;
x3=8;
x4=-9;
o1=subs(h1)
tt=abs(o1)/6
j=1;


for t=0:0.0002:10
 z(1,j)=x1 
 z(2,j)=x2
 z(3,j)=x3
 z(4,j)=x4
 for i=1:3
     a(i)=subs(g(i));
 end;
 
b1=subs(h1);

k=1;
d=sqrt((x1-x10)^2+(x2-x20)^2+(x3-x30)^2+(x4-x40)^2);

 m(1)=subs(diff(f,['x',num2str(1)]));
 m(2)=subs(diff(f,['x',num2str(2)]));
 m(3)=subs(diff(f,['x',num2str(3)]));
 m(4)=subs(diff(f,['x',num2str(4)]));
 
 m=[m(1);m(2);m(3);m(4)];
mm=norm(m);

p1=0;
p2=0;
p3=0;
p4=0;

for i=1:3
if (a(i)>0)
    n(i)=subs(diff(g(i),['x',num2str(1)]));
    nn(i)=subs(diff(g(i),['x',num2str(2)]));
    nnn(i)=subs(diff(g(i),['x',num2str(3)]));
    nnnn(i)=subs(diff(g(i),['x',num2str(4)]));   
elseif(a(i)==0)
    n(i)=0.5*subs(diff(g(i),['x',num2str(1)]));
    nn(i)=0.5*subs(diff(g(i),['x',num2str(2)]));
    nnn(i)=0.5*subs(diff(g(i),['x',num2str(3)]));
    nnnn(i)=0.5*subs(diff(g(i),['x',num2str(4)])); 
else
    n(i)=0;
    nn(i)=0;
    nnn(i)=0;
    nnnn(i)=0;  
end;
p1=n(i)+p1;
p2=nn(i)+p2;
p3=nnn(i)+p3;
p4=nnnn(i)+p4;
end
p=[p1;p2;p3;p4];
if (b1>0)
    hh1=1;
elseif (b1<0)
    hh1=-1;
else
    hh1=0.5;
end;
if (t>tt)
    u=1;
else
    u=0;
end


s=-u*(I-P)*(m+((mm/k)*d+a0)*p)-A'*hh1 
  x1=x1+l*s(1);
  x2=x2+l*s(2); 
  x3=x3+l*s(3); 
  x4=x4+l*s(4); 
 j=j+1     
end

t=0:0.0002:10;
plot(t,z(1,:),t,z(2,:),t,z(3,:),t,z(4,:))
xlabel('time(sec)');
ylabel('state variables');
%title('状态向量x(t)的运动轨迹图形');
set(get(gca,'xlabel'),'FontSize',10);
set(get(gca,'ylabel'),'FontSize',10);
%set(get(gca,'title'),'FontSize',10);
h=legend('x1','x2','x3','x4',1);
hold off;
