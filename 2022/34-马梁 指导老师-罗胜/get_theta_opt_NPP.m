function [theta] = get_theta_opt_NPP(alpha,epsilon,w)
%GET_THETA_OPT Summary of this function goes here
%   Detailed explanation goes here


load("generate_channel.mat","N")

lambda=ones(N,1);%%%%%%%%%%%%%%如何求λ_opt?;
d=0;
for i=1:N
    e=zeros(N,1);
    e(i)=1;
    d=d+lambda(i).*e*e';

end

theta=(get_v_U(epsilon,w)+d)\get_v_v(alpha,epsilon,w);
end

