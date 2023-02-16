%{
This is the training code for MUSER algorithm.

Input:
    data(MLAI structure):
        data:arranged by column

Output:
    result:W,Q,U,P,funcvalue

Related paper:
Partial Multi-Label Learning via Multi-Subspace Representation

Author: coffee
Date: 2022.11.29
%}

function [result] = MUSER_train(data, pars)
    if data.infm.sample_order == 'row'
        X = data.data';
    else
        X = data.data;
    end

    X = normalize(X,'range');
    if isfield(pars,'algo') && pars.algo == 1
        algo = 1;   % use the method of mine
    else
        algo = 0;  % use the optimization of paper
    end

    n = data.infm.nr_sample;
    q = data.infm.nr_class;
    d = data.infm.dim;

    Y = pars.Y;
    Alpha = pars.Alpha;
    Beta = pars.Beta;
    Gamma = pars.Gamma;
    Lambda_U = pars.lr_rate;
    Lambda_Q = pars.lr_rate;
    c = pars.c;
    m = pars.m;
    L = pars.L;

    P = randn(c,q);
    U = randn(n,c);
    Q = randn(d,m);
    W = randn(m,c);
    cvg = false;
    T = 1;
    fun(T) = Count_fun(W,Q,U,P,data,pars);

    while ~cvg && T<pars.T_max
        if algo == 0
            P = (Alpha*(U'*U)+Gamma*eye(c))\(Alpha*U'*Y);
            det_U = (1+Gamma)*U+Beta*L*U+Alpha*U*(P*P')-Alpha*Y*P'-X'*Q*W;
            U = U - Lambda_U*det_U;
            Q = Q - Lambda_Q*(-X*U*W'+X*X'*Q*(W*W'));
            tmp = Q * ones(m,1);
            Q = Q./tmp;
            W = (Q'*X*(Q'*X)'+Gamma*eye(m))\Q'*X*U;
        else
            P = (Alpha*(U'*U)+Gamma*eye(c))\(Alpha*U'*Y);
%             U = sylvester((1+Gamma)*eye(n)-Beta*L,Alpha*(P*P'),Alpha*Y*P'+X'*Q*W);
            det_U = (1+Gamma)*U+Beta*L*U+Alpha*U*(P*P')-Alpha*Y*P'-X'*Q*W;
            U = U - Lambda_U*det_U;
            W = ((X'*Q)'*X'*Q + 1e-5*eye(m))\Q'*X*U;
            H = (X*X'+1e-5*eye(d))\(X*U*(X*U)');
            H = (H+H')/2;
            [EV, D] = eig(H);
            [~, ind] = sort(diag(D),'descend');
            EVs = EV(:,ind);
            Q = EVs(:,1:m);
        end

        T = T + 1;
        fun(T) = Count_fun(W,Q,U,P,data,pars);
        if (abs(fun(T) - fun(T-1))/fun(T) < 1e-6)
            cvg = true;
        end
    end
    result.W = W;
    result.Q = Q;
    result.U = U;
    result.P = P;
    result.funcvalue = fun;
end
function fun = Count_fun(W,Q,U,P,data,pars)
    if data.infm.sample_order == 'row'
        X = data.data';
    else
        X = data.data;
    end

    fun = 1/2*norm(U-X'*Q*W,'fro')^2+pars.Alpha/2*norm(data.label-U*P,'fro')^2+ ...
        pars.Beta/2*trace(U'*pars.L*U)+pars.Gamma/2*(norm(W,'fro')^2+ ...
        norm(U,'fro')^2+norm(P,'fro')^2);
end