function [Exactly,condition,citr,slst,X, f, v, Pbestx, Pbestf,b,state] = Global_search1(Data,bu,bd,condition,slst,n,termination,citr,X, f, v, Pbestx, Pbestf, Gbestx, Gbestf)
c = size(Data,2)-1;
% kernal='cubic';
kernal='linear';

Bd = repmat(bd,c,1); Bu = repmat(bu,c,1);
    L = Data(:,1:c); LY = Data(:,c+1);
    global coefC
    coefC = rbfcreate(L',LY','RBFFunction', kernal);
    nameR = @YCRBF;
evals = size(Data,1); bound = [bd, bu]; state = zeros(size(slst,2),1);
for i = 1 : size(slst,2)
    if condition(i) == 0 
        [f, X, Pbestx, Pbestf, Gbestx, Gbestf, v, citr, state] = PSO_search(X, evals, slst, f, v, Pbestx, Pbestf, Gbestx, Gbestf, i, bound, citr, n,termination,nameR,state);
    end
end
P = Gbestx; index = find(state~=1); pf = Gbestf;
[~,b] = min(pf); 
Exactly = P;
Exactly(find(state~=1),:) = [];
end