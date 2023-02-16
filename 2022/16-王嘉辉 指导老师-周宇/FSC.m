function [prodata,remdata,weights]=FSC(data,label)

%使用ReliefF函数作为分类器对特征值进行分类
[ranks,weights] = relieff(data,label,1,'method','classification');

%确定膝点
maxpoint = [1,weights(ranks(1))];
minpoint = [size(ranks,2),weights(ranks(size(ranks,2)))];
maxDistance = 0;
kneepoint = 0;
for i=1:size(ranks,2)
    point = [i,weights(ranks(i))];
    d = abs(det([minpoint-maxpoint;point-maxpoint]))/norm(minpoint-maxpoint);
    if d > maxDistance
        maxDistance = d;
        kneepoint = i;
    end
end

%将原本的特征集分为高权值特征集和低权值特征集
for i=1:size(ranks,2)
    if i <= kneepoint
        prodata(i) = ranks(i);
    else
        remdata(i-kneepoint) = ranks(i);
    end
end

end