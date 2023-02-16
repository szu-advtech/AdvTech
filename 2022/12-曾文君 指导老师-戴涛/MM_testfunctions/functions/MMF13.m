function y=MMF13(x)
% 0.1<=x1<=1.1    0.1<=x2<=1.1  0.1<=x3<=1.1
% 1global PS and 1 local PS, PSs are non-liner 





    y=zeros(2,1);
    g=2-exp(-2*log10(2)*((x(2)+sqrt(x(3))-0.1)/0.8).^2)*(sin(2*pi*(x(2)+sqrt(x(3)))))^6;
    y(1)=x(1);
    y(2)=g/x(1);
% %%
end

% Multi-objective Genetic Algorithms: Problem Difficulties and Construction of Test Problems
