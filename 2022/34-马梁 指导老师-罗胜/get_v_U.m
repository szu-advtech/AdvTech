function [U] = get_v_U(epsilon,w)
%GET_V_U Summary of this function goes here
%   Detailed explanation goes here

e=0;
f=0;

load("generate_channel.mat","K")
for k=1:K

    for i=1:K
        f=f+get_a_ik(i,k,w)*get_a_ik(i,k,w)';
    end
    e=e+abs(epsilon(k))^2*f;
    f=0;

end


U=e;
end

