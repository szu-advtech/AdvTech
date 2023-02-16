
function y=SYM_PART_simple(x)
%-20<=xi<=20
%Reference[Rudolph G, Naujoks B, Preuss M. Capabilities of EMOA to detect and preserve equivalent Pareto subsets[C]//Evolutionary Multi-Criterion Optimization. Springer Berlin/Heidelberg, 2007: 36-50.]
a=1;
b=10;
c=8;

temp_t1=sign(x(1))*ceil((abs(x(1))-(a+c/2))/(2*a+c));
temp_t2=sign(x(2))*ceil((abs(x(2))-b/2)/b);
t1=sign(temp_t1)*min(abs(temp_t1),1);
t2=sign(temp_t2)*min(abs(temp_t2),1);

x1=x(1)-t1*(c+2*a);
x2=x(2)-t2*b;
y=fun([x1,x2],a);


end


function y=fun(x,a)
y=zeros(2,1);
y(1)=(x(1)+a)^2+x(2)^2;
y(2)=(x(1)-a)^2+x(2)^2;

end




