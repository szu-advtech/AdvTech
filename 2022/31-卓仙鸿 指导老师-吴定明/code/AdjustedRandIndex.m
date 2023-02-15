% 出处 https://github.com/kunzhan/GSF
% 作者个人主页 https://github.com/kunzhan
function ARI = AdjustedRandIndex(groundtruth, cluster_res)
% 参考https://zhuanlan.zhihu.com/p/161703182

% 创建列联矩阵（表），得到的这个矩阵（表）和二分图是一样的
% 列联矩阵（表）反映实例类别划分与聚类划分的重叠程度
% 表的行表示实际划分的类别，表的列表示聚类划分的簇标记，nij表示重叠实例数量
ContingencyTable=zeros(max(groundtruth),max(cluster_res));

for i = 1:length(groundtruth)
   ContingencyTable(groundtruth(i),cluster_res(i))=ContingencyTable(groundtruth(i),cluster_res(i))+1;
end

n = sum(sum(ContingencyTable));                     % C中所有元素求和
n_i_square = sum(sum(ContingencyTable,2).^2);		% 每行求和后再平方，将每一行平方后的值再加起来
n_j_square = sum(sum(ContingencyTable,1).^2);		% 每列求和后再平方，将每一列平方后的值再加起来

a = (n *(n-1)) / 2;		   
b = sum(sum(ContingencyTable.^2));	                % 矩阵每个元素平方后，再求和
c = 0.5*(n_i_square + n_j_square);           

A = a + b - c;		                                %no. agreements 也就是index
Expected_index = (n*(n^2+1)-(n+1)*n_i_square-(n+1)*n_j_square+2*(n_i_square*n_j_square)/n)/(2*(n-1));

ARI=(A-Expected_index)/(a-Expected_index);




