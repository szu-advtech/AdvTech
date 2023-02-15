

% ��ȡ���е�.mat�ļ�

clear
resPath='D:\�ҵ��ĵ�\MATLAB\myCodeFinal\';

matPath{1}=[resPath,'Run_N���\'];
matPath{2}=[resPath,'Run_M3T���\'];
matPath{3}=[resPath,'Run_Proposed���\'];
% matPath{..}=..   % ������ѡ��ܶ��ļ���(mat)

fileNum=length(matPath);

for i=1:fileNum
    allRes{i}=dir([matPath{i},'\*.mat']);
    subLength(i)=length(allRes{i});
end

matNum=unique(subLength);

% �ж��ļ��������mat�ļ������Ƿ�һ��
if(length(matNum)~=1) 
    sprintf('����ÿ���ļ��������mat�ļ�������һ��!\n');
    return;
end


select='Res1';      % ѡ�����Fscore
% select='Res2';    % ѡ�����Acc

for i=1:fileNum % fileNum
    temp=0;
    for j=1:matNum
        
        row=(i-1)*matNum+j;
        strName=allRes{i}(j).name;
        
        load ([matPath{i},strName]) % ���ζ������ļ���ȡmat

        Res=eval(select); % ѡ�����Acc

        
        a(row,1)=Res.svmAcc; % a �Ƕ�ȡ�ĵ�ǰmat��һ��
        a(row,2)=Res.svmSen;
        a(row,3)=Res.svmSpec;
        a(row,4)=Res.svmPrec;
        a(row,5)=Res.svmFscore;
        a(row,6)=Res.svmAuc;
        a(row,7)=Res.maxAcc;
        a(row,8)=Res.minArmse;
        a(row,9)=Res.maxBcc;
        a(row,10)=Res.minBrmse;
        
        b(i+temp,:)=a(row,:); % b �����ǰ�������Ҫ�Ĺ��ɻ�ȡ
        matName{i+temp,1}=strName;
        temp=temp+3;
        
    end
end

b=real(b);

titleName={'Feature','ACC','SEN','SPEC','PREC','Fscore','AUC','scoreAcc','ARMSE','scoreBcc','BRMSE'};

combine=[matName,num2cell(b)];
combine=[titleName;combine];

% xlsFile �����xls�ļ�·����

% if(exist(xlsFile,'file')) % ����Ѿ����ڣ���ô����ɾ��.
%    delete(xlsFile);
% end


xlsFile=['c:\all_',select,'_Compare.xlsx'];
xlswrite(xlsFile, combine); % ��need_xml_infoֱ��ת����xls��ʽ


