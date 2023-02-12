function [w_k] = max_w_k(w,beta_k_opt,alpha,Theta)
%MAX_W_K Summary of this function goes here
%   Detailed explanation goes here


load("generate_channel.mat","P_T","K")
lambda=0.0;
w_k=zeros(K);
n=P_T+1;
    while n>P_T
        if lambda>10^3
            lambda=lambda*2;
        end

        if(n>2*P_T)
            lambda=2*lambda;
        end
        lambda=lambda+0.01;
        for k=1:K
            w_k(:,k)=get_w_k_opt(k,beta_k_opt,lambda,alpha,Theta);
        end
        n=0;
        for i=1:K
            n=n+norm(w_k(:,i),2);
        end
        if lambda>=10
            break;
        end
    end

end

