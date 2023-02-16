function [w] = get_w_k_opt(k,beta_k,lambda,alpha,Theta)
%GET_W_K_OPT Summary of this function goes here
%   Detailed explanation goes here
load("generate_channel.mat","omiga","K","sigma","M","h_rk","h_dk","G");
den=0;
for i=1:K
    den=den+(norm(beta_k(i))^2)*...
        (h_dk(:,i)+G'*Theta*h_rk(:,i))*(h_dk(:,i)+G'*Theta*h_rk(:,i))';
end


w=sqrt(omiga(k)*(1+alpha(k)))*beta_k(k)*(lambda.*eye(M)+den)^(-1)*(h_dk(:,k)+G'*Theta*h_rk(:,k));

end

