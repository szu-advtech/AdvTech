function f = Omni_test(x)
% 0<=xi<= 6  
   f=zeros(2,1);
   n=length(x);
   for i=1:n
       f(1)=f(1)+sin(pi*x(i));
       f(2)=f(2)+cos(pi*x(i));
   end
  
end

%%
%Reference Omni-Optimizer: A Procedure for Single and Multi-Objective
%          Optimization
%          Approximating the Set of Pareto Optimal Solutions in Both the Decision and Objective Spaces by an Estimation of Distribution Algorithm



