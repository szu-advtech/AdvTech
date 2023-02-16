function [C] = get_v_c(alpha,epsilon,w)
%GET_V_C Summary of this function goes here

load("generate_channel.mat","K","omiga","sigma");

C=0;
for k=1:K
    t=0;
    for i=1:K
        t=t+norm(get_b_ik(i,k,w))^2;
    end

    C=C+2*sqrt(omiga(k)*(1+alpha(k)))*real(epsilon(k)*get_b_ik(k,k,w)) ...
        -abs(epsilon(k))^2*(sigma^2+t);
end
end




