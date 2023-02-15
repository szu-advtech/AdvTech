function [theta] = get_theta_ADMM(alpha,epsilon,w)
%GET_THETA_A Summary of this function goes here
%   Detailed explanation goes here
load("generate_channel.mat","N");


t=ones(N,1);
t_=zeros(N,1);
q=ones(N,1);
l=ones(N,1);
miu=get_miu(epsilon,w);
while abs(norm(t)-norm(t_))>0.001
    t_=t;
    t=admm_theta(q,t,l,miu);
    
    q=admm_q(t,q,l,alpha,miu,epsilon,w);
    l=admm_lam(t,q,l,miu);
end
theta=t; 
end


function[t_] = admm_theta(q,t,l,miu)
    t_=q-1/miu*l;
    t_=t_/norm(t_)*min(1,norm(t_));
end
function[q_] = admm_q(q,t,l,alpha,miu,epsilon,w)
    load("generate_channel.mat","N");
    q_=(2*get_v_U(epsilon,w)+miu*eye(N))^(-1)*(2*get_v_v(alpha,epsilon,w)+l+miu*t);
end
function[l_] = admm_lam(q,t,l,miu)
    l_=l-miu*(q-t);
end


function[r] = posdef(mat)
    d=eig(mat);
    if all(d>0)
        r=1;
        return
    end
        r=0;
end

function[miu]=get_miu(epsilon,w)
load("generate_channel.mat","N");

    miu=norm(get_v_U(epsilon,w));
    tao=1;
    while posdef(tao*miu/2.*eye(N)-get_v_U(epsilon,w))
        tao=tao+1;
    end
    miu=miu*tao;
end

