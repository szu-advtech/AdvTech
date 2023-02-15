function [theta] = get_theta_ICU(alpha,epsilon,w, t)
%GET_THETA_ Summary of this function goes here
%   Detailed explanation goes here


load("generate_channel.mat","N");
theta=zeros(N,1);
t_=zeros(N,1);
U=get_v_U(epsilon,w);
v=get_v_v(alpha,epsilon,w);
while t~=t_
    t_=t;
    for n=1:N
        d=0;
        for j=1:N
            if j~=n
                d=d+U(n,j)*t(j);
            end
        end
        theta(n)=(v(n)-d)/abs(v(n)-d)*min(1,abs(v(n)-d)/U(n,n));
    end
end
end
