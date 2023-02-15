function y=MMF10(x)
% 0.1<x1<=1.1    0.1<=x2<=1.1   g(x2)>0
% 1global PS and 1 local PS 

% f1(x1,x2)=x1;
% f2(x1,x2)=g(x2)/x1;

  y=zeros(2,1);
 
    g=2-exp(-((x(2)-0.2)/0.004).^2)-0.8*exp(-((x(2)-0.6)/0.4).^2);
    y(1)=x(1);
    y(2)=g/x(1);
% %%
end

% Multi-objective Genetic Algorithms: Problem Difficulties and Construction of Test Problems
