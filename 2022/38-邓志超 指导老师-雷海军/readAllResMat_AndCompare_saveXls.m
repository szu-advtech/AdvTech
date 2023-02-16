

% 获取所有的.mat文件

clear
resPath='D:\我的文档\MATLAB\myCodeFinal\';

matPath{1}=[resPath,'Run_N结果\'];
matPath{2}=[resPath,'Run_M3T结果\'];
matPath{3}=[resPath,'Run_Proposed结果\'];
% matPath{..}=..   % 还可以选择很多文件夹(mat)

fileNum=length(matPath);

for i=1:fileNum
    allRes{i}=dir([matPath{i},'\*.mat']);
    subLength(i)=length(allRes{i});
end

matNum=unique(subLength);

% 判断文件夹下面的mat文件个数是否一致
if(length(matNum)~=1) 
    sprintf('错误：每个文件夹下面的mat文件个数不一致!\n');
    return;
end


select='Res1';      % 选择最大Fscore
% select='Res2';    % 选择最大Acc

for i=1:fileNum % fileNum
    temp=0;
    for j=1:matNum
        
        row=(i-1)*matNum+j;
        strName=allRes{i}(j).name;
        
        load ([matPath{i},strName]) % 依次读各个文件夹取mat

        Res=eval(select); % 选择最大Acc

        
        a(row,1)=Res.svmAcc; % a 是读取的当前mat的一行
        a(row,2)=Res.svmSen;
        a(row,3)=Res.svmSpec;
        a(row,4)=Res.svmPrec;
        a(row,5)=Res.svmFscore;
        a(row,6)=Res.svmAuc;
        a(row,7)=Res.maxAcc;
        a(row,8)=Res.minArmse;
        a(row,9)=Res.maxBcc;
        a(row,10)=Res.minBrmse;
        
        b(i+temp,:)=a(row,:); % b 矩阵是按我们需要的规律获取
        matName{i+temp,1}=strName;
        temp=temp+3;
        
    end
end

b=real(b);

titleName={'Feature','ACC','SEN','SPEC','PREC','Fscore','AUC','scoreAcc','ARMSE','scoreBcc','BRMSE'};

combine=[matName,num2cell(b)];
combine=[titleName;combine];

% xlsFile 保存的xls文件路径。

% if(exist(xlsFile,'file')) % 如果已经存在，那么就先删除.
%    delete(xlsFile);
% end


xlsFile=['c:\all_',select,'_Compare.xlsx'];
xlswrite(xlsFile, combine); % 将need_xml_info直接转换成xls格式


