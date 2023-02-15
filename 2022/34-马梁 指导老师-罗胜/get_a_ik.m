function [a] = get_a_ik(i,k,w)

load("generate_channel.mat","eta","h_rk","G");

a=sqrt(eta)*diag(h_rk(:,k)')*G*w(:,i);
end

