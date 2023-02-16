% 出处 https://github.com/kunzhan/GSF
% 作者个人主页 https://github.com/kunzhan
function [ACC,NMI,Purity,Precision,Recall,F_score,ARI] = metric_compute(groundtruth, cluster_res)
% precision = TP / (TP + FP)
% recall = TP / (TP + FN)
% F-score = (2 * precision * recall) / (precision + recall)
% RI = (TP + TN) / (TP + FP + FN +TN)
% https://en.wikipedia.org/wiki/Confusion_matrix
n = length(groundtruth);
numOfClass = length(unique(groundtruth));   % 分类个数

% -------------------------Precision,Recall,F_score-----------------------
TP = 0;
TP_plus_FP = 0; % (TP + FP)
TP_plus_FN = 0; % (TP + FN)

for i=1:n
    % 下面两条语句是因为groundtruth和cluster_res的"顺序"反了过来。因此不会有"遗漏"
    cluster_res_i = (cluster_res(i+1:end)) == cluster_res(i);
    groundtruth_i = (groundtruth(i+1:end)) == groundtruth(i);
    
    TP = TP + sum(groundtruth_i .* cluster_res_i);
    TP_plus_FP = TP_plus_FP + sum(cluster_res_i);
    TP_plus_FN = TP_plus_FN + sum(groundtruth_i);
end

Precision = TP / TP_plus_FP;
Recall = TP / TP_plus_FN;

if (Precision + Recall) == 0
   F_score = 0;
else
   F_score = 2 * Precision * Recall / (Precision + Recall);
end
% -----------------------Precision,Recall,F_score--------------------------
% https://en.wikipedia.org/wiki/Rand_index
ARI = AdjustedRandIndex(groundtruth, cluster_res);

% ------------------------ACC,MI,Purity,p-value----------------------------
% purity：https://zhuanlan.zhihu.com/p/53840697
cluster_res_index = unique(cluster_res); % 算法分类结果的类别名
max_occurrence = 0;                      % 某个分类出现次数最多的那个元素的个数
for i = 1:numOfClass
    
    incluster = groundtruth(find(cluster_res == cluster_res_index(i)));
    cluster_state = hist(incluster, 1:max(incluster));
    % incluster是一个列向量，由hist方法得到的inclunub是一个行向量，大多数元素为0
    % cluster_state的每个非0分量表示在第i个聚类中，某一类元素出现的次数。每个非0元素代表1类，
    % 例如，这是循环到某次时，cluster_state的结果为：0 0 72 0 0 68 0 0 0 0 0 0 0 0 0 0 0 0 62
    % 这表示，该分类中有3种元素，72个第3类元素，68个第6类元素，62个第19类元素，但它们被分到一个类去了（算法分错了）
    % 根据purity的定义，总是要取每个分量数量最多的那种，即max(cluster_state)
    % 所以，68和62就舍弃掉了，这就导致purity不能达到100%
    % 理想状态下，每次循环得到的inclunub应该只有一个非0元素（分类正确）。
    if isempty(cluster_state) 
        cluster_state=0;
    end
    max_occurrence = max_occurrence + max(cluster_state);
end
Purity = max_occurrence/ n;

% 用匈牙利算法对groundtruth向量和分类结果向量进行最大匹配
% groundtruth向量和分类结果向量可以看成是两个图
% 这么做是为了计算ACC和MI
res_after_map = hungarian_Map(groundtruth, cluster_res);

ACC = length(find(res_after_map == groundtruth)) / length(groundtruth);

% NMI:https://zhuanlan.zhihu.com/p/53840697
% NMI:https://zh.wikipedia.org/wiki/%E4%BA%92%E4%BF%A1%E6%81%AF
NMI = MutualInfo(groundtruth,res_after_map);

[p,h] = signtest(groundtruth, res_after_map)    % 论文5.3部分的non-parametric pair-wised Wilcoxon test
% ---------------------------ACC,MI,Purity--------------------------------



% ------------------------------------------------------------------------
function newRes = hungarian_Map(groundtruth,cluster_res)
% 用匈牙利算法对groundtruth向量和分类结果向量进行最大匹配
% 构造二分图，便于等下使用匈牙利算法得到最大匹配
numOfClass = length(unique(groundtruth));
Bipartition_Graph = zeros(numOfClass);
for i=1:numOfClass
    for j=1:numOfClass
        Bipartition_Graph(i,j) = length(find(groundtruth == i & cluster_res == j));
    end
end
% 得到的二分图矩阵Bipartition_Graph的第i列是第i次的cluster_state。矩阵描述了分类情况！
% 理想情况下矩阵Bipartition_Graph是一个对角阵，对角元素为左下到右上
[bestPermute,~] = hungarian(-Bipartition_Graph);      % 用匈牙利算法得到最优排列
newRes = zeros(numOfClass,1);                         % 初始化最终结果
for i=1:numOfClass
    newRes(cluster_res == i) = bestPermute(i);
end




function NMI = MutualInfo(groundtruth,cluster_res)

numOfClass = length(unique(groundtruth));

% 构造二分图
Bipartition_Graph = zeros(numOfClass);
for i=1:numOfClass
    for j=1:numOfClass
        Bipartition_Graph(i,j) = length(find(groundtruth == i & cluster_res == j));
    end
end
% 现在得到的Bipartition_Graph和hungarian_Map方法中的Bipartition_Graph对角线方向反过来了！
% 得到的Bipartition_Graph可以看成是联合概率分布（未归一化）

sumG = sum(Bipartition_Graph(:));

% 接下来计算互信息
% 行的边缘分布p1，结果是列向量
P1 = sum(Bipartition_Graph,2);  
P1 = P1/sumG;

% 列的边缘分布p2，结果是行向量
P2 = sum(Bipartition_Graph,1);  
P2 = P2/sumG;

% 计算边缘分布的熵，熵的计算方式类似于期望(本质就是期望)
% 参考信息论
H1 = sum(-P1.*log2(P1));
H2 = sum(-P2.*log2(P2));

P12 = Bipartition_Graph/sumG;   % 联合概率分布
aver = P12./repmat(P2,numOfClass,1)./repmat(P1,1,numOfClass);
aver(abs(aver) < 1e-12) = 1; 
MI = sum(P12(:) .* log2(aver(:)));

NMI = MI / max(H1,H2);          % 归一化
% ----------------------------------------------------------------------


% ----------------------------匈牙利算法-----------------------------------
% 指派问题的匈牙利算法参考下面链接：
% https://blog.csdn.net/Dark_Scope/article/details/8880547
% https://blog.csdn.net/u013384984/article/details/90718287
function [C,T]=hungarian(A)
%HUNGARIAN Solve the Assignment problem using the Hungarian method.
%
%[C,T]=hungarian(A)
%A - a square cost matrix.
%C - the optimal assignment.
%T - the cost of the optimal assignment.
%s.t. T = trace(A(C,:)) is minimized over all possible assignments.

% Adapted from the FORTRAN IV code in Carpaneto and Toth, "Algorithm 548:
% Solution of the assignment problem [H]", ACM Transactions on
% Mathematical Software, 6(1):104-111, 1980.

% v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
%                 Department of Computing Science, Ume?University,
%                 Sweden. 
%                 All standard disclaimers apply.

% A substantial effort was put into this code. If you use it for a
% publication or otherwise, please include an acknowledgement or at least
% notify me by email. /Niclas

[m,n]=size(A);

if (m~=n)
    error('HUNGARIAN: Cost matrix must be square!');
end

% Save original cost matrix.
orig=A;

% Reduce matrix.
A=hminired(A);

% Do an initial assignment.
[A,C,U]=hminiass(A);

% Repeat while we have unassigned rows.
while (U(n+1))
    % Start with no path, no unchecked zeros, and no unexplored rows.
    LR=zeros(1,n);
    LC=zeros(1,n);
    CH=zeros(1,n);
    RH=[zeros(1,n) -1];
    
    % No labelled columns.
    SLC=[];
    
    % Start path in first unassigned row.
    r=U(n+1);
    % Mark row with end-of-path label.
    LR(r)=-1;
    % Insert row first in labelled row set.
    SLR=r;
    
    % Repeat until we manage to find an assignable zero.
    while (1)
        % If there are free zeros in row r
        if (A(r,n+1)~=0)
            % ...get column of first free zero.
            l=-A(r,n+1);
            
            % If there are more free zeros in row r and row r in not
            % yet marked as unexplored..
            if (A(r,l)~=0 & RH(r)==0)
                % Insert row r first in unexplored list.
                RH(r)=RH(n+1);
                RH(n+1)=r;
                
                % Mark in which column the next unexplored zero in this row
                % is.
                CH(r)=-A(r,l);
            end
        else
            % If all rows are explored..
            if (RH(n+1)<=0)
                % Reduce matrix.
                [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
            end
            
            % Re-start with first unexplored row.
            r=RH(n+1);
            % Get column of next free zero in row r.
            l=CH(r);
            % Advance "column of next free zero".
            CH(r)=-A(r,l);
            % If this zero is last in the list..
            if (A(r,l)==0)
                % ...remove row r from unexplored list.
                RH(n+1)=RH(r);
                RH(r)=0;
            end
        end
        
        % While the column l is labelled, i.e. in path.
        while (LC(l)~=0)
            % If row r is explored..
            if (RH(r)==0)
                % If all rows are explored..
                if (RH(n+1)<=0)
                    % Reduce cost matrix.
                    [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
                end
                
                % Re-start with first unexplored row.
                r=RH(n+1);
            end
            
            % Get column of next free zero in row r.
            l=CH(r);
            
            % Advance "column of next free zero".
            CH(r)=-A(r,l);
            
            % If this zero is last in list..
            if(A(r,l)==0)
                % ...remove row r from unexplored list.
                RH(n+1)=RH(r);
                RH(r)=0;
            end
        end
        
        % If the column found is unassigned..
        if (C(l)==0)
            % Flip all zeros along the path in LR,LC.
            [A,C,U]=hmflip(A,C,LC,LR,U,l,r);
            % ...and exit to continue with next unassigned row.
            break;
        else
            % ...else add zero to path.
            
            % Label column l with row r.
            LC(l)=r;
            
            % Add l to the set of labelled columns.
            SLC=[SLC l];
            
            % Continue with the row assigned to column l.
            r=C(l);
            
            % Label row r with column l.
            LR(r)=l;
            
            % Add r to the set of labelled rows.
            SLR=[SLR r];
        end
    end
end

% Calculate the total cost.
T=sum(orig(logical(sparse(C,1:size(orig,2),1))));


function A=hminired(A)
%HMINIRED Initial reduction of cost matrix for the Hungarian method.
%
%B=assredin(A)
%A - the unreduced cost matris.
%B - the reduced cost matrix with linked zeros in each row.

% v1.0  96-06-13. Niclas Borlin, niclas@cs.umu.se.

[m,n]=size(A);

% Subtract column-minimum values from each column.
colMin=min(A);
A=A-colMin(ones(n,1),:);

% Subtract row-minimum values from each row.
rowMin=min(A')';
A=A-rowMin(:,ones(1,n));

% Get positions of all zeros.
[i,j]=find(A==0);

% Extend A to give room for row zero list header column.
A(1,n+1)=0;
for k=1:n
    % Get all column in this row. 
    cols=j(k==i)';
    % Insert pointers in matrix.
    A(k,[n+1 cols])=[-cols 0];
end


function [A,C,U]=hminiass(A)
%HMINIASS Initial assignment of the Hungarian method.
%
%[B,C,U]=hminiass(A)
%A - the reduced cost matrix.
%B - the reduced cost matrix, with assigned zeros removed from lists.
%C - a vector. C(J)=I means row I is assigned to column J,
%              i.e. there is an assigned zero in position I,J.
%U - a vector with a linked list of unassigned rows.

% v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

[n,np1]=size(A);

% Initalize return vectors.
C=zeros(1,n);
U=zeros(1,n+1);

% Initialize last/next zero "pointers".
LZ=zeros(1,n);
NZ=zeros(1,n);

for i=1:n
    % Set j to first unassigned zero in row i.
	lj=n+1;
	j=-A(i,lj);

    % Repeat until we have no more zeros (j==0) or we find a zero
	% in an unassigned column (c(j)==0).
    
	while (C(j)~=0)
		% Advance lj and j in zero list.
		lj=j;
		j=-A(i,lj);
	
		% Stop if we hit end of list.
		if (j==0)
			break;
		end
	end

	if (j~=0)
		% We found a zero in an unassigned column.
		
		% Assign row i to column j.
		C(j)=i;
		
		% Remove A(i,j) from unassigned zero list.
		A(i,lj)=A(i,j);

		% Update next/last unassigned zero pointers.
		NZ(i)=-A(i,j);
		LZ(i)=lj;

		% Indicate A(i,j) is an assigned zero.
		A(i,j)=0;
	else
		% We found no zero in an unassigned column.

		% Check all zeros in this row.

		lj=n+1;
		j=-A(i,lj);
		
		% Check all zeros in this row for a suitable zero in another row.
		while (j~=0)
			% Check the in the row assigned to this column.
			r=C(j);
			
			% Pick up last/next pointers.
			lm=LZ(r);
			m=NZ(r);
			
			% Check all unchecked zeros in free list of this row.
			while (m~=0)
				% Stop if we find an unassigned column.
				if (C(m)==0)
					break;
				end
				
				% Advance one step in list.
				lm=m;
				m=-A(r,lm);
			end
			
			if (m==0)
				% We failed on row r. Continue with next zero on row i.
				lj=j;
				j=-A(i,lj);
			else
				% We found a zero in an unassigned column.
			
				% Replace zero at (r,m) in unassigned list with zero at (r,j)
				A(r,lm)=-j;
				A(r,j)=A(r,m);
			
				% Update last/next pointers in row r.
				NZ(r)=-A(r,m);
				LZ(r)=j;
			
				% Mark A(r,m) as an assigned zero in the matrix . . .
				A(r,m)=0;
			
				% ...and in the assignment vector.
				C(m)=r;
			
				% Remove A(i,j) from unassigned list.
				A(i,lj)=A(i,j);
			
				% Update last/next pointers in row r.
				NZ(i)=-A(i,j);
				LZ(i)=lj;
			
				% Mark A(r,m) as an assigned zero in the matrix . . .
				A(i,j)=0;
			
				% ...and in the assignment vector.
				C(j)=i;
				
				% Stop search.
				break;
			end
		end
	end
end

% Create vector with list of unassigned rows.

% Mark all rows have assignment.
r=zeros(1,n);
rows=C(C~=0);
r(rows)=rows;
empty=find(r==0);

% Create vector with linked list of unassigned rows.
U=zeros(1,n+1);
U([n+1 empty])=[empty 0];


function [A,C,U]=hmflip(A,C,LC,LR,U,l,r)
%HMFLIP Flip assignment state of all zeros along a path.
%
%[A,C,U]=hmflip(A,C,LC,LR,U,l,r)
%Input:
%A   - the cost matrix.
%C   - the assignment vector.
%LC  - the column label vector.
%LR  - the row label vector.
%U   - the 
%r,l - position of last zero in path.
%Output:
%A   - updated cost matrix.
%C   - updated assignment vector.
%U   - updated unassigned row list vector.

% v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

n=size(A,1);

while (1)
    % Move assignment in column l to row r.
    C(l)=r;
    
    % Find zero to be removed from zero list..
    
    % Find zero before this.
    m=find(A(r,:)==-l);
    
    % Link past this zero.
    A(r,m)=A(r,l);
    
    A(r,l)=0;
    
    % If this was the first zero of the path..
    if (LR(r)<0)
        ...remove row from unassigned row list and return.
        U(n+1)=U(r);
        U(r)=0;
        return;
    else
        
        % Move back in this row along the path and get column of next zero.
        l=LR(r);
        
        % Insert zero at (r,l) first in zero list.
        A(r,l)=A(r,n+1);
        A(r,n+1)=-l;
        
        % Continue back along the column to get row of next zero in path.
        r=LC(l);
    end
end


function [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
%HMREDUCE Reduce parts of cost matrix in the Hungerian method.
%
%[A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
%Input:
%A   - Cost matrix.
%CH  - vector of column of 'next zeros' in each row.
%RH  - vector with list of unexplored rows.
%LC  - column labels.
%RC  - row labels.
%SLC - set of column labels.
%SLR - set of row labels.
%
%Output:
%A   - Reduced cost matrix.
%CH  - Updated vector of 'next zeros' in each row.
%RH  - Updated vector of unexplored rows.

% v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

n=size(A,1);

% Find which rows are covered, i.e. unlabelled.
coveredRows=LR==0;

% Find which columns are covered, i.e. labelled.
coveredCols=LC~=0;

r=find(~coveredRows);
c=find(~coveredCols);

% Get minimum of uncovered elements.
m=min(min(A(r,c)));

% Subtract minimum from all uncovered elements.
A(r,c)=A(r,c)-m;

% Check all uncovered columns..
for j=c
    % ...and uncovered rows in path order..
    for i=SLR
        % If this is a (new) zero..
        if (A(i,j)==0)
            % If the row is not in unexplored list..
            if (RH(i)==0)
                % ...insert it first in unexplored list.
                RH(i)=RH(n+1);
                RH(n+1)=i;
                % Mark this zero as "next free" in this row.
                CH(i)=j;
            end
            % Find last unassigned zero on row I.
            row=A(i,:);
            colsInList=-row(row<0);
            if (length(colsInList)==0)
                % No zeros in the list.
                l=n+1;
            else
                l=colsInList(row(colsInList)==0);
            end
            % Append this zero to end of list.
            A(i,l)=-j;
        end
    end
end

% Add minimum to all doubly covered elements.
r=find(coveredRows);
c=find(coveredCols);

% Take care of the zeros we will remove.
[i,j]=find(A(r,c)<=0);

i=r(i);
j=c(j);

for k=1:length(i)
    % Find zero before this in this row.
    lj=find(A(i(k),:)==-j(k));
    % Link past it.
    A(i(k),lj)=A(i(k),j(k));
    % Mark it as assigned.
    A(i(k),j(k))=0;
end

A(r,c)=A(r,c)+m;



