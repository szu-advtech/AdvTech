function [beta_k_opt] = get_beta_k_opt(k,w,Theta,alpha)
%GET_BETA_K_OPT Summary of this function goes here
%   Detailed explanation goes here


load("generate_channel.mat","omiga","K","sigma","h_dk","h_rk","G");
den=0;
for i=1:K
    den=den+norm((h_dk(:,k)'+G'*Theta*h_rk(:,k))'*w(:,i))^2;
end


beta_k_opt=sqrt(omiga(k)*(1+alpha))*(h_dk(:,k)+G'*Theta*h_rk(:,k))'*w(:,k)/(den+sigma^2);
end

