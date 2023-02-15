
% Excel格式要求如下：

% Feature	ACC     SEN	S   PEC     PREC	Fscore	AUC     scoreAcc	ARMSE	scoreBcc	BRMSE
% 1.1.Mat	0.737 	0.363 	0.909 	0.622 	0.440 	0.780 	0.411       3.385 	0.466       8.081 
% 1.2.Mat_	0.743 	0.357 	0.921 	0.783 	0.443 	0.717 	0.395       3.350 	0.451       8.012 

% 比较长度与最短(行数最少)的为准


clear 

mainPath='D:\实验结果\';


xlsFile{1}=[mainPath,'N_1Cross_Res1.xlsx'];
xlsFile{2}=[mainPath,'Lasso_1Cross_Res1.xlsx'];
xlsFile{3}=[mainPath,'M3T_1Cross_Res1.xlsx'];
xlsFile{4}=[mainPath,'PRO_1Cross_folder1_Res1.xlsx'];
% xlsFile{..}=..  % 还可以添加更多

saveFile=[mainPath,'compareAll.xlsx'];


xlsFilesNum=length(xlsFile);


for i=1:xlsFilesNum
    [~,~,xlsContent{i}]=xlsread(xlsFile{i});
    title{i}=xlsContent{i}(1,:);
    xlsSubContent{i}=xlsContent{i}(2:end,:);
    
    xlsFilesRows(i)=size(xlsSubContent{i},1);
    xlsFilesColumn(i)=size(xlsSubContent{i},2);

end


% 首先判断行数是否一致，不一致，直接结束。
lenRow=unique(xlsFilesRows); % 如果行数一致的话，那么只有一个唯一的数字
isRowEqual=length(unique(xlsFilesRows)); 
if(isRowEqual~=1)
    fprintf('%s\n','错误：有Excel行数不一致!');
    return;
end

% 判断列数是否一致，不一致，直接结束。
lenColumn=unique(xlsFilesColumn); % 如果行数一致的话，那么只有一个唯一的数字
isColumnEqual=length(unique(xlsFilesColumn)); 
if(isColumnEqual~=1)
    fprintf('%s\n','错误：有Excel列数不一致!');
    return;
end

% 运行致此，说明行数、列数是一致的


% xlsSubContent
k=1;
compareCell=cell(xlsFilesNum*lenRow,lenColumn); % xlsFilesNum表示：总共的xls文件个数
% 进行合并
for i=1:lenRow
    for j=1:xlsFilesNum
        compareCell(k,:)=xlsSubContent{j}(i,:);
        k=k+1; % 每两行空一行
    end
    k=k+1;
end



compareCell=[title{1};compareCell];

xlswrite(saveFile,compareCell); % 将need_xml_info直接转换成xls格式




