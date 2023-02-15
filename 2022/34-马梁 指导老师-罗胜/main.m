
lambda=0;

T=14;
re=zeros(1,T);
k=0;
P_T=1;
for t=1:T
    
    %init_system;
    load("generate_channel.mat","G","h_rk","h_dk","N","K","M","sigma","omiga", "eta");

    P_T=P_T+0.3;
    save("generate_channel.mat","G","h_rk","h_dk","N","K","M","sigma","omiga", "P_T","eta");
    gama=ones(K,1);
    beta_k=ones(K,1);
    epsilon=ones(K,1);
    w=ones(M,K);
    theta=ones(N,1);

    Theta=diag(theta);

    a=-9;
    b=-8;
    while (a-b)<0.1
        b=a;
        for i=1:K
            gama(i)=get_gama(i,w,Theta);
        end

        for i=1:K
            beta_k(i)=get_beta_k_opt(i,w,Theta,gama(i));
        end

        lambda=0;

        for i=1:K
            w(:,i)=get_w_k_opt(i,beta_k,lambda,gama,Theta);
        end
        lambda=0.01;
        w=max_w_k(w,beta_k,gama,Theta);

        for i=1:K
           epsilon(i)=get_epsilon_opt(i,theta,gama(i),w);
        end


%           theta=ones(N,1); %%没有IRS的baseline
%        theta=get_theta_opt_NPP(gama,epsilon,w);
%         theta=get_theta_ICU(gama,epsilon,w,theta);
%            theta=get_theta_ADMM(gama,epsilon,w);
        [a,theta]=max_f4_cvx(gama,w,epsilon);
        for i=1:N
%           theta(i)=project_to_F2(theta(i));  %%投影到F2
        end
        Theta=diag(theta);
        a=f1a(w,Theta,gama)
        c=f1(w,Theta)
%         U=get_v_U(epsilon,w);
%         v=get_v_v(gama,epsilon,w);
%         a=-((U^(1/2)*theta)'*(U^(1/2)*theta))+2*real(theta'*v)+get_v_c(alpha,epsilon,w)
    end
    re(t)=a;
    
end
legend('F2','No IRS','F1');
xlabel('P_T');
ylabel('R');
plot([1:1:14],re);
re

