% 出处 https://github.com/kunzhan/GSF
% 作者个人主页 https://github.com/kunzhan
function distance = Euclidean_distance(A,B)
% 欧几里得距离为||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
% 在GSF中，A == B
% dist是一个方阵，返回的是向量之间的距离。参考论文即知
% dist方阵的每一个元素表示两个向量的欧氏距离的平方
% 例如，A和B都是m*n的矩阵，则dist第i行第j列表示A的第i列与B的第j列的欧氏距离的平方
% 例如，A = [1,3,5;2,4,6]; B = [2,6,3;5,1,7];
% dist = [10,26,29;2,18,9;10,26,5];  (1-3)^2 + (2-7)^2 = 29

col = size(A, 2);
dist_size = col;
distance = zeros(dist_size, dist_size);

% for i = 1:col
%     for j = 1:col
%         distance(i, j) = norm(A(:,i) - B(:,j)) * norm(A(:,i) - B(:,j));
%     end
% end

A_square_sum = sum(A.*A);
B_square_sum = sum(B.*B);

A_square = repmat(A_square_sum',[1 size(B_square_sum,2)]);%||A||^2
B_square = repmat(B_square_sum,[size(A_square_sum,2) 1]); %||B||^2
AB=A'*B; 
distance = A_square + B_square - 2*AB;












