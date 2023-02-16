function [condition, slst, Gbestx, Gbestf] = condition_detectation(X, condition, slst, Roverlap, Rexcl, Gbestx, Gbestf, Pbestf)
for i = 1: size(slst,2)
    a = size(slst{i},2);
    if condition(i) == 0
        center1 = mean(X(slst{i},:),1);
        for j = 1: size(slst,2)
            if i ~= j && condition(j) == 0
                center2 = mean(X(slst{j},:),1);
                r1 = ((sum((repmat(center1,size(slst{j},2),1) - X(slst{j},:)).^2,2)).^0.5)';
                rover1 = r1 <= Rexcl;
                r2 = ((sum((repmat(center2,size(slst{i},2),1) - X(slst{i},:)).^2,2)).^0.5)';
                rover2 = r2 <= Rexcl;
                rover = min(sum(rover1)/size(slst{j},2), sum(rover2)/size(slst{i},2));
                if rover >= Roverlap
                    condition(j) = 2;
                    slst{i} = [slst{i},slst{j}];
                    if Gbestf(i) > Gbestf(j)
                        Gbestf(i) = Gbestf(j);
                        Gbestx(i,:) = Gbestx(j,:);
                    end
                    if size(slst{i},2)> a%n(2)
                        [~,b] = sort(Pbestf(slst{i}));
                        slst{j} = slst{i}(b(a+1:end));
                        slst{i} = slst{i}(b(1:a));
                    end
                end
            end
        end
    end
end

end