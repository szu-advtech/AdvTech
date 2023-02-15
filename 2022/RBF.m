function [ W2,B2,Centers,Spreads ] = RBF( SamIn,SamOut,Nc)
% Usage: [ W2,B2,Centers,Spreads ] = RBF( SamIn,SamOut,Nc)
% Build a Single RBF Model Using the Method below.
% K.-L. Du and M. Swamy, “Radial basis function networks,” in Neural Networks and Statistical Learning.   Springer, 2014, pp. 299–335.
% Input:
% SamIn         - Offline Data with c Decision Variables
% SamOut        - Offline Data with Exact Objective Value
% Nc            - Number of neurons of RBF models

%
% Output: 
% W2             - Weights of RBF Model
% B2             - Bais of RBF Model
% Centers        - Centers of RBF Model
% Spreads        - Widths of RBF model
%------------------------------------------------------------------------
SamIn=SamIn';
SamOut=SamOut';
SamNum = size(SamIn,2); 
InDim = size(SamIn,1); 
ClusterNum = Nc; 
Overlap = 1.0; 

A=randperm(SamNum);
% 随机选取中心
Centers = SamIn(:,A(1:ClusterNum));

ik=0;
while 1
    NumberInClusters = zeros(ClusterNum,1); 
    IndexInClusters = zeros(ClusterNum,SamNum);

% 计算样本与中心的距离，对样本进行分类
    for i = 1:SamNum
        AllDistance = dist(Centers',SamIn(:,i));
        [MinDist,Pos] = min(AllDistance);
        NumberInClusters(Pos) = NumberInClusters(Pos) + 1;
        IndexInClusters(Pos,NumberInClusters(Pos)) = i;
    end
    OldCenters = Centers;

    for i = 1:ClusterNum
        if NumberInClusters(i)~=0
            Index = IndexInClusters(i,1:NumberInClusters(i));
            Centers(:,i) = mean(SamIn(:,Index),2);
        else
            A=randperm(SamNum);
            Centers(:,i)=SamIn(:,A(1));
        end
    end
    EqualNum = sum(sum(Centers==OldCenters));
    if EqualNum == InDim*ClusterNum|ik>=50
        break;
    end
    ik=ik+1;
end

AllDistances = dist(Centers',Centers); 
Maximum = max(max(AllDistances)); 
for i = 1:ClusterNum 
AllDistances(i,i) = Maximum+1;
end
I=find(AllDistances==0);
AllDistances(I)=0.000001;
Spreads = Overlap*min(AllDistances)'; 


Distance = dist(Centers',SamIn); 
SpreadsMat = repmat(Spreads,1,SamNum);
HiddenUnitOut = radbas(Distance./SpreadsMat); 
HiddenUnitOutEx = [HiddenUnitOut' ones(SamNum,1)]'; 
W2Ex = SamOut*pinv(HiddenUnitOutEx); 
W2 = W2Ex(:,1:ClusterNum); 
B2 = W2Ex(:,ClusterNum+1); 


end

