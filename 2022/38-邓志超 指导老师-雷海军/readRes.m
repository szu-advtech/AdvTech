

% 获取所有的.mat文件

clear
resPath=cd;
allRes=dir([resPath,'\*.mat']);


len=length(allRes);


% select='Res1';      % 选择最大Fscore
select='Res2';    % 选择最大Acc

for i=1:len
    strName{i}=allRes(i).name;
    load (strName{i}); % 依次读取mat

    Res=eval(select); % 选择最大Acc
    
    a(i,1)=Res.svmAcc;
    a(i,2)=Res.svmSen;
    a(i,3)=Res.svmSpec;
    a(i,4)=Res.svmPrec;
    a(i,5)=Res.svmFscore;
    a(i,6)=Res.svmAuc;
    a(i,7)=Res.maxAcc;
    a(i,8)=Res.minArmse;
    a(i,9)=Res.maxBcc;
    a(i,10)=Res.minBrmse;
end


a=real(a);
matName=strName';
titleName={'Feature','ACC','SEN','SPEC','PREC','Fscore','AUC','scoreAcc','ARMSE','scoreBcc','BRMSE'};

combine=[matName,num2cell(a)];
combine=[titleName;combine];

% xlsFile 保存的xls文件路径。

% if(exist(xlsFile,'file')) % 如果已经存在，那么就先删除.
%    delete(xlsFile);
% end
xlsFile=['c:\',select,'.xlsx'];
xlswrite(xlsFile, combine); % 将need_xml_info直接转换成xls格式




