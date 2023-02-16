function [epsilon] = get_epsilon_opt(k,theta,alpha,w)
%GET_EPSILON_OPT get ε;
%   Detailed explanation goes here
load("generate_channel.mat","K","sigma","h_dk","eta","G","omiga","h_rk");

c=0;
den=0;
for i=1:K
    den=den+abs(get_b_ik(i,k,w)+theta'*(get_a_ik(i,k,w)))^2;

end

epsilon=sqrt(omiga(k)*(1+alpha))*(get_b_ik(k,k,w)+theta'*get_a_ik(i,k,w))/(den+sigma^2);
end

