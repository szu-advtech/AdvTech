function [r,th]=max_f4_cvx(alpha,w, epsilon)

load("generate_channel.mat","N");
n = N;
U=get_v_U(epsilon,w);
v=get_v_v(alpha,epsilon,w);

cvx_begin SDP
    variable theta(n,1) complex
    variable t
    maximize t
    subject to
    t+((U^(1/2)*theta)'*(U^(1/2)*theta))-2*real(theta'*v)-get_v_c(alpha,epsilon,w)<=0;
    for i=1:n    
        theta(i)*theta(i)' <= 1;
    end
cvx_end

th=theta;
r=t;