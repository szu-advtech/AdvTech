
% Excel��ʽҪ�����£�

% Feature	ACC     SEN	S   PEC     PREC	Fscore	AUC     scoreAcc	ARMSE	scoreBcc	BRMSE
% 1.1.Mat	0.737 	0.363 	0.909 	0.622 	0.440 	0.780 	0.411       3.385 	0.466       8.081 
% 1.2.Mat_	0.743 	0.357 	0.921 	0.783 	0.443 	0.717 	0.395       3.350 	0.451       8.012 

% �Ƚϳ��������(��������)��Ϊ׼


clear 

mainPath='D:\ʵ����\';


xlsFile{1}=[mainPath,'N_1Cross_Res1.xlsx'];
xlsFile{2}=[mainPath,'Lasso_1Cross_Res1.xlsx'];
xlsFile{3}=[mainPath,'M3T_1Cross_Res1.xlsx'];
xlsFile{4}=[mainPath,'PRO_1Cross_folder1_Res1.xlsx'];
% xlsFile{..}=..  % ��������Ӹ���

saveFile=[mainPath,'compareAll.xlsx'];


xlsFilesNum=length(xlsFile);


for i=1:xlsFilesNum
    [~,~,xlsContent{i}]=xlsread(xlsFile{i});
    title{i}=xlsContent{i}(1,:);
    xlsSubContent{i}=xlsContent{i}(2:end,:);
    
    xlsFilesRows(i)=size(xlsSubContent{i},1);
    xlsFilesColumn(i)=size(xlsSubContent{i},2);

end


% �����ж������Ƿ�һ�£���һ�£�ֱ�ӽ�����
lenRow=unique(xlsFilesRows); % �������һ�µĻ�����ôֻ��һ��Ψһ������
isRowEqual=length(unique(xlsFilesRows)); 
if(isRowEqual~=1)
    fprintf('%s\n','������Excel������һ��!');
    return;
end

% �ж������Ƿ�һ�£���һ�£�ֱ�ӽ�����
lenColumn=unique(xlsFilesColumn); % �������һ�µĻ�����ôֻ��һ��Ψһ������
isColumnEqual=length(unique(xlsFilesColumn)); 
if(isColumnEqual~=1)
    fprintf('%s\n','������Excel������һ��!');
    return;
end

% �����´ˣ�˵��������������һ�µ�


% xlsSubContent
k=1;
compareCell=cell(xlsFilesNum*lenRow,lenColumn); % xlsFilesNum��ʾ���ܹ���xls�ļ�����
% ���кϲ�
for i=1:lenRow
    for j=1:xlsFilesNum
        compareCell(k,:)=xlsSubContent{j}(i,:);
        k=k+1; % ÿ���п�һ��
    end
    k=k+1;
end



compareCell=[title{1};compareCell];

xlswrite(saveFile,compareCell); % ��need_xml_infoֱ��ת����xls��ʽ




