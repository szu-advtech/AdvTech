function [gama] = get_gama(k,w,Theta)
%得到信干噪比gama
%   第k个用户的信干噪比

load("generate_channel.mat","G","K","M","N","sigma","h_dk","h_rk");

count=0;
t=0;

for x=1:K
    if x~=k
        t=t+norm((h_dk(:,k)'+h_rk(:,k)'*Theta'*G)*w(:,x))^2;
    end
end




gama=(norm((h_dk(:,k)'+h_rk(:,k)'*Theta'*G)*w(:,k)))^2/(t+sigma^2);
end
