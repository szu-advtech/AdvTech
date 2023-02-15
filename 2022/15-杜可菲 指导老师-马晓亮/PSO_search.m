function [f, X, Pbestx, Pbestf, Gbestx, Gbestf, v, citr,state] = PSO_search(X, evals, slst, f, v, Pbestx, Pbestf, Gbestx, Gbestf, k, bound, citr, n,termination,name,state)
[~ ,dim] = size(X);
for i = 1:size(slst{k},2)
    v(slst{k}(i),:) = (0.6-0.3*citr(k)/((termination-evals)/n(1))).*v(slst{k}(i),:) + 1.7*rand(1, size(v,2)).*(Pbestx(slst{k}(i),:) - X(slst{k}(i),:)) + 1.7*rand(1, size(v,2)).*(Gbestx(k,:) - X(slst{k}(i),:));
    v(v<bound(1)) = bound(1);v(v>bound(2)) = bound(2);
    currentx = X(slst{k}(i),:);
    X(slst{k}(i),:) = X(slst{k}(i),:) + v(slst{k}(i),:);
    for j = 1:dim
        if X(slst{k}(i),j) < bound(1)
            X(slst{k}(i),j) = bound(1) + rand*(bound(2)-bound(1));
        else if X(slst{k}(i),j) > bound(2)
                X(slst{k}(i),j) = bound(2) - rand*(bound(2)-bound(1));
            end
        end
    end
    g(i)=feval(name,X(slst{k}(i),:));
    f(slst{k}(i)) = g(i);
    citr(k) = (citr(k)+1)/(n(1));
    
    if g(i) < Pbestf(slst{k}(i))
        Pbestf(slst{k}(i)) = g(i);
        Pbestx(slst{k}(i),:) = X(slst{k}(i),:);
        for j = 1:size(X,2)
            a = Gbestx(k,:);
            a(j) = Pbestx(slst{k}(i),j);
            b = feval(name,a);
            if b < Gbestf(k)
                Gbestf(k) = b;
                Gbestx(k,:) = a;
                state(k) = 1;
            end
        end
    end
 
end
end