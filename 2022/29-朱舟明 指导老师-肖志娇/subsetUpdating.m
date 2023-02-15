%获得更新的子集（变异）
function [subset] = subsetUpdating(subset,scaleFactor)

    numSelect = sum(subset);
    numChange = floor(numSelect * scaleFactor);
    selectIndex = find(subset==1);
    notSelectIndex = find(subset==0);
    noSelect = selectIndex(randperm(numel(selectIndex),numChange));
    
    select = notSelectIndex(randperm(numel(notSelectIndex),numChange));
    subset(noSelect) = false;
    subset(select) = true;
end

