function [v_v] = get_v_v(alpha,epsilon,w)
%GET_V_V Summary of this function goes here

load("generate_channel.mat","K","omiga")
v_v=0;
for k=1:K

    a=0;
    for i=1:K
        a=a+get_b_ik(i,k,w)'*get_a_ik(i,k,w);

    end
    v_v=v_v+sqrt(omiga(k)*(1+alpha(k)))*epsilon(k)'*get_a_ik(k,k,w)-abs(epsilon(k))^2*a;
end

end

